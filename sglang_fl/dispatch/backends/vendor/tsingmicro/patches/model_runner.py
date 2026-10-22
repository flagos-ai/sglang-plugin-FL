# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch SGLang's ``ModelRunner.forward()`` to add per‑iteration
performance timing.

Equivalent to VLLM's ``VLLM_FL_TIMER`` feature in ``WorkerFL.execute_model()``:
records wall‑clock time (including device synchronize) around the full
``forward()`` call and prints a one‑line summary on rank 0.

Controlled by ``SGLANG_FL_TIMER_ENABLE`` environment variable (or the legacy
``VLLM_FL_TIMER_ENABLE`` for backward compatibility).  Set to ``1`` to activate.

Output format (one line per forward pass on rank 0)::

    [INFO][SGLANG_FL_TIMER] iter=12 tp_rank=0 seq_len=8192 time=45.6789ms

Patches applied:
  1. ``ModelRunner.__init__`` — inject ``_fl_timer_enabled`` flag (post‑init hook).
  2. ``ModelRunner.forward`` — wrap with sync → timer → sync → print.
"""

import functools
import os
import time

import torch

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger

_log = patch_logger("model_runner")

_patched = False


def _is_enabled() -> bool:
    """Check whether FL timer is enabled via environment variable."""
    return (
        os.environ.get("SGLANG_FL_TIMER_ENABLE", "0") == "1"
        or os.environ.get("VLLM_FL_TIMER_ENABLE", "0") == "1"
    )


def _platform_synchronize() -> None:
    """Device synchronize — works cross‑platform.

    On txda, ``torch_txda`` remaps ``torch.cuda.synchronize()`` to
    ``torch.txda.synchronize()`` transparently.  On other platforms it
    calls the native synchronize.
    """
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        # Device not initialized or no CUDA context — safe to ignore
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def patch() -> None:
    """Apply model-runner timing monkey-patches.

    Idempotent: safe to call multiple times.  No‑op when
    ``SGLANG_FL_TIMER_ENABLE`` is not ``1``.
    """
    global _patched
    if _patched:
        return

    if not _is_enabled():
        _log.skipped(
            "SGLANG_FL_TIMER_ENABLE not set — model_runner timer skipped"
        )
        _patched = True
        return

    try:
        from sglang.srt.model_executor.model_runner import ModelRunner
    except Exception as exc:
        _log.failed("failed to import ModelRunner: %s", exc)
        return

    # ── 1. Inject timer flag into __init__ ──────────────────────────────────

    _orig_init = ModelRunner.__init__

    @functools.wraps(_orig_init)
    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self._fl_timer_enabled = _is_enabled()

    ModelRunner.__init__ = _patched_init

    # ── 2. Wrap forward() with timer ────────────────────────────────────────

    _orig_forward = ModelRunner.forward

    @functools.wraps(_orig_forward)
    def _patched_forward(self, forward_batch, *args, **kwargs):
        if not getattr(self, "_fl_timer_enabled", False) or self.tp_rank != 0:
            return _orig_forward(self, forward_batch, *args, **kwargs)

        # Iteration counter — ModelRunner.forward() increments
        # forward_pass_id inside the original.  We read it beforehand
        # and add 1 so the logged number reflects *this* iteration.
        iter_num = self.forward_pass_id + 1

        _platform_synchronize()
        t0 = time.perf_counter()

        result = _orig_forward(self, forward_batch, *args, **kwargs)

        _platform_synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Extract the number of tokens inferred this step and the forward mode
        # from the batch (NOT from the logits output, whose first dim is
        # batch_size and would mislabel prefill as 1).
        num_tokens = _extract_num_tokens(forward_batch)
        mode = _extract_forward_mode(forward_batch)

        print(
            f"[INFO][SGLANG_FL_TIMER] iter={iter_num} "
            f"tp_rank={self.tp_rank} "
            f"mode={mode} "
            f"seq_len={num_tokens} "
            f"time={elapsed_ms:.4f}ms",
            flush=True,
        )

        return result

    ModelRunner.forward = _patched_forward

    _patched = True
    _log.applied("ModelRunner.forward patched with FL timer")


def _extract_num_tokens(forward_batch) -> int:
    """Number of tokens inferred in *this* forward step (not cumulative).

    * decode: ``batch_size`` — each request decodes exactly one token.
    * prefill / extend / mixed: ``extend_num_tokens`` — the newly prefilled
      tokens for this step.
    * fallback: ``seq_lens_sum``.

    Note ``seq_lens_sum`` is the *sum of full sequence lengths* (KV-cache
    tokens), which grows by one every decode step and is therefore NOT what we
    want here.
    """
    fm = getattr(forward_batch, "forward_mode", None)
    if fm is not None:
        try:
            if fm.is_decode():
                return int(getattr(forward_batch, "batch_size", 0))
        except Exception:
            pass

    nt = getattr(forward_batch, "extend_num_tokens", None)
    if nt is not None:
        try:
            return int(nt)
        except Exception:
            pass

    return int(getattr(forward_batch, "seq_lens_sum", 0))


def _extract_forward_mode(forward_batch) -> str:
    """Human-readable forward mode, mapped to ``prefill`` / ``decode``.

    SGLang's ``ForwardMode`` exposes ``is_decode()`` and ``is_prefill()``
    (== ``is_extend()``, covering EXTEND / MIXED / DRAFT_EXTEND /
    TARGET_VERIFY / SPLIT_PREFILL / DLLM_EXTEND).  Anything else (IDLE,
    PREBUILT, …) falls back to the lower-cased enum name.
    """
    fm = getattr(forward_batch, "forward_mode", None)
    if fm is None:
        return "unknown"
    try:
        if fm.is_decode():
            return "decode"
        if fm.is_prefill():
            return "prefill"
    except Exception:
        pass
    return getattr(fm, "name", "unknown").lower()
