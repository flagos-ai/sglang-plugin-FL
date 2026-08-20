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

        # Extract seq_len from the forward output.
        seq_len = _extract_seq_len(result, forward_batch)

        print(
            f"[INFO][SGLANG_FL_TIMER] iter={iter_num} "
            f"tp_rank={self.tp_rank} "
            f"seq_len={seq_len} "
            f"time={elapsed_ms:.4f}ms",
            flush=True,
        )

        return result

    ModelRunner.forward = _patched_forward

    _patched = True
    _log.applied("ModelRunner.forward patched with FL timer")


def _extract_seq_len(result, forward_batch) -> int:
    """Best-effort extraction of sequence length from forward output."""
    try:
        logits_out = result.logits_output
    except AttributeError:
        return getattr(forward_batch, "seq_lens_sum", 0)

    if logits_out is None:
        return getattr(forward_batch, "seq_lens_sum", 0)

    for attr in ("hidden_states", "next_token_logits"):
        val = getattr(logits_out, attr, None)
        if val is not None:
            return val.shape[0]

    return getattr(forward_batch, "seq_lens_sum", 0)
