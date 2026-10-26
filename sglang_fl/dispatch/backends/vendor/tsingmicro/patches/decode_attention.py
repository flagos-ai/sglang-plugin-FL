# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route SGLang decode attention to the FlagGems txda two-segment kernel.

SGLang's decode attention reads the KV cache through CSR layout -- ``kv_indptr``
(cumulative lengths) plus a flattened ``kv_indices`` of physical token slots --
over a flat ``[pool_size, kv_heads, head_dim]`` pool.  FlagGems' txda
``decode_attention_fwd_sglang`` implements exactly that contract with a
two-segment kernel: for each request the leading run of *physically contiguous*
slots (the prompt, as the page_size=1 allocator laid it out) is walked affinely
(``slot = kv_indices[base] + offset``), and the rest is handed to a masked gather.
That avoids the token-by-token gather SGLang's triton kernel does, and was
measured ~23x faster at the 2k decode scenario (84 ms vs 1.9 s per layer).

``prefix_lens`` (how long each request's contiguous run is) is derived inside
the FlagGems op by ``_compute_prefix_lens`` / ``_prefix_len_kernel`` whenever
``prefix_lens=None`` is passed: a native TXDA scan, no host sync.  Computing it
here in Python instead used to go through FlagGems' *generic* ``cumprod`` (no
``_tsingmicro`` specialization), which lowers to ``tts.get_structured_state`` and
faults the device (HPGR error 0x600000F).  This patch therefore only installs
when the installed FlagGems op supports the in-kernel scan.

Note that over-estimating ``prefix_lens`` is silent: the kernel would read
physically contiguous slots that belong to another request and return plausible
but wrong logits, with no crash.  Keep the merge step of that scan off FlagGems'
generic ``min_dim`` -- see ``_TXDA_EXCLUDED_ATEN_OPS`` in ``sglang_fl/__init__.py``.

Falls back to SGLang's original kernel when the installed FlagGems lacks the op,
lacks the in-kernel prefix scan, or a request uses a feature the op does not
model (sinks, XAI temperature, ``v_scale != 1``, KV dtype != activation dtype).

Gated by ``SGLANG_FL_DECODE_ATTN_FLAGOS`` (default ``1`` = enabled; set ``0`` to
disable).
"""

import inspect
import os
import sys

import torch

from sglang_fl.dispatch.backends.vendor.tsingmicro.flaggems_compat import (
    load_txda_op,
)
from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger

_log = patch_logger("decode_attention")

_patched = False
_originals = {}
# One-shot per-process diagnostics (decode attention runs per layer, per step).
_logged = {"routed": False, "feature_fallback": False}

_ENABLED_KEY = "SGLANG_FL_DECODE_ATTN_FLAGOS"
_OP_NAME = "decode_attention_fwd_sglang"


def _enabled() -> bool:
    return os.getenv(_ENABLED_KEY, "1").lower() in ("1", "true", "yes", "on")


def _supports_in_kernel_prefix(fn) -> bool:
    """Whether ``fn`` accepts ``prefix_lens=None`` and derives it on device.

    Older FlagGems releases shipped ``decode_attention_fwd_sglang`` with a
    mandatory ``prefix_lens``; passing ``None`` there either raises or -- worse --
    is read as an empty prefix.  ``_compute_prefix_lens`` is the marker of the
    version that scans in-kernel, so require both.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # builtins / opaque wrappers
        return False
    if "prefix_lens" not in params:
        return False
    module = sys.modules.get(getattr(fn, "__module__", ""), None)
    return getattr(module, "_compute_prefix_lens", None) is not None


def _call_original(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    k_scale,
    v_scale,
    logit_cap,
    sinks,
    xai_temperature_len,
):
    """Call SGLang's ``decode_attention_fwd`` with its original signature."""
    return _originals["decode_attention_fwd"](
        q,
        k_buffer,
        v_buffer,
        o,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        k_scale,
        v_scale,
        logit_cap=logit_cap,
        sinks=sinks,
        xai_temperature_len=xai_temperature_len,
    )


def _fl_decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    k_scale,
    v_scale,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
):
    """FlagGems-backed drop-in for ``decode_attention_fwd``."""
    # Features the FlagGems op does not model → original kernel.
    if (
        sinks is not None
        or xai_temperature_len > 0
        or v_scale != 1.0
        or k_buffer.dtype != q.dtype
        or v_buffer.dtype != q.dtype
    ):
        if not _logged["feature_fallback"]:
            _logged["feature_fallback"] = True
            _log.warning(
                "decode_attention FALLBACK: unsupported feature "
                "(sinks=%s xai=%d v_scale=%s k_dtype=%s v_dtype=%s)",
                sinks is not None, xai_temperature_len, v_scale,
                k_buffer.dtype, v_buffer.dtype,
            )
        return _call_original(
            q, k_buffer, v_buffer, o, kv_indptr, kv_indices, attn_logits, attn_lse,
            num_kv_splits, max_kv_splits, sm_scale, k_scale, v_scale,
            logit_cap, sinks, xai_temperature_len,
        )

    txda_decode_attention_fwd_sglang = load_txda_op(_OP_NAME)
    if txda_decode_attention_fwd_sglang is None:  # pragma: no cover - patch() guards
        return _call_original(
            q, k_buffer, v_buffer, o, kv_indptr, kv_indices, attn_logits, attn_lse,
            num_kv_splits, max_kv_splits, sm_scale, k_scale, v_scale,
            logit_cap, sinks, xai_temperature_len,
        )

    batch = q.shape[0]
    head_num = q.shape[1]

    # attn_logits/attn_lse (SGLang's split-K scratch) are unused: the FlagGems op
    # runs its own stage-1/stage-2 split internally and writes straight to ``o``.
    # Its lse output is not consumed by SGLang, so a throwaway buffer is enough.
    fl_lse = torch.empty(batch, head_num, dtype=torch.float32, device=q.device)

    # k_scale is folded into the softmax scale (SGLang does the same).
    txda_decode_attention_fwd_sglang(
        q,
        k_buffer,
        v_buffer,
        o,
        fl_lse,
        kv_indptr,
        kv_indices,
        None,  # prefix_lens: computed in-kernel from kv_indptr/kv_indices
        sm_scale * k_scale,
        logit_cap,
    )

    if not _logged["routed"]:
        _logged["routed"] = True
        _log.info(
            "decode_attention -> flag_gems two-segment (batch=%d, prefix_lens in-kernel)",
            batch,
        )


def patch() -> None:
    """Monkey-patch SGLang's ``decode_attention_fwd`` to use FlagGems. Idempotent."""
    global _patched
    if _patched:
        return
    if not _enabled():
        _log.skipped("disabled (set %s=1 to enable)", _ENABLED_KEY)
        return

    # Older flag_gems releases ship the _tsingmicro backend without this op (or
    # without the in-kernel prefix scan); keep SGLang's own kernel in that case.
    fn = load_txda_op(_OP_NAME)
    if fn is None:
        _log.skipped("flag_gems has no %s", _OP_NAME)
        return
    if not _supports_in_kernel_prefix(fn):
        _log.skipped("flag_gems %s cannot derive prefix_lens in-kernel", _OP_NAME)
        return

    try:
        import sglang.srt.layers.attention.triton_ops.decode_attention as mod
    except Exception as exc:  # noqa: BLE001
        _log.failed("failed to import sglang decode_attention: %s", exc)
        return

    _originals["decode_attention_fwd"] = mod.decode_attention_fwd
    mod.decode_attention_fwd = _fl_decode_attention_fwd
    _patched = True
    _log.applied("decode_attention_fwd routed to flag_gems two-segment kernel")
