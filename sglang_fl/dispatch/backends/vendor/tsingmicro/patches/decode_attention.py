# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route SGLang decode attention to the FlagGems txda blockptr fast path.

SGLang's decode attention reads the KV cache through CSR layout — ``kv_indptr``
(cumulative lengths) plus a flattened ``kv_indices`` of physical token slots —
over a flat ``[pool_size, kv_heads, head_dim]`` pool.  FlagGems'
``decode_attention_fwd`` uses vLLM's paged layout and its txda blockptr kernel,
which is ~23x faster than the token-by-token masked gather at the 2k decode
scenario (84 ms vs 1.9 s per layer).

SGLang's flat pool is contiguous per request (page_size=1 allocator), so it can
be reinterpreted as a paged layout with ``page_size=16`` *provided every
request's physical slots are 16-aligned* (true when prompts are a multiple of 16
tokens, as in the Qwen3.6 offline profile).  This patch converts CSR→dense, checks
contiguity + 16-alignment in one device reduction, and:

* if aligned → FlagGems blockptr (fast path);
* otherwise (radix-cache shared prefixes, non-aligned starts, sink tokens, XAI
  temperature, fp8/scaled KV) → SGLang's original kernel unchanged.

Gated by ``SGLANG_FL_DECODE_ATTN_FLAGOS`` (default ``1`` = enabled; set ``0`` to
disable).
"""

import os

import torch

from sglang_fl.dispatch.backends.vendor.tsingmicro.flaggems_compat import (
    load_txda_op,
)
from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger

_log = patch_logger("decode_attention")

_patched = False
_originals = {}
# One-shot per-process diagnostics (decode attention runs per layer, per step).
_logged = {"routed": False, "feature_fallback": False, "align_fallback": False}

_ENABLED_KEY = "SGLANG_FL_DECODE_ATTN_FLAGOS"
_PAGE_SIZE = 16

# txda exposes 16 tiles/SMs (see the flag_gems decode_attention docstring).
# SGLang reports device_core_count=0 on txda, which disables its own adaptive
# split-K logic and makes every request inherit ``max_kv_splits``.
_TXDA_TILES = 16


def _enabled() -> bool:
    return os.getenv(_ENABLED_KEY, "1").lower() in ("1", "true", "yes", "on")


def _choose_num_kv_splits(batch, head_num, kv_heads, max_kv_splits):
    """Pick a split-K count that keeps the 16 txda tiles occupied.

    Split-K only pays off when one program per (request, head-group) would leave
    the device idle.  For batch >= 4 the base grid already fills the tiles, so any
    split just adds the stage-2 reduction overhead (measured ~2x slower at 2k).
    """
    if max_kv_splits <= 1:
        return 1
    kv_group = max(1, head_num // kv_heads)
    if kv_group == 1:
        head_groups = head_num  # MHA normal kernel: one program per head
    else:
        block_h = min(16, kv_group)  # mirrors _decode_grouped_kernel_stage1
        head_groups = (head_num + block_h - 1) // block_h
    base = max(1, batch * head_groups)
    needed = (_TXDA_TILES + base - 1) // base
    return max(1, min(max_kv_splits, needed))


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
    original = _originals["decode_attention_fwd"]

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
        return original(
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

    txda_decode_attention_fwd_sglang = load_txda_op("decode_attention_fwd_sglang")

    batch = q.shape[0]
    device = q.device

    # prefix_lens (leading contiguous run per request = the prompt) is computed
    # inside the FlagGems op via _compute_prefix_lens/_prefix_len_kernel (a
    # native TXDA scan, no host sync).  Computing it here in Python with
    # torch.cumprod used to hit FlagGems' *generic* cumprod (no _tsingmicro
    # specialization), which lowers to tts.get_structured_state and faults the
    # device (HPGR device error 0x600000F).  Pass None to use the in-kernel scan.
    head_num = q.shape[1]
    fl_lse = torch.empty(batch, head_num, dtype=torch.float32, device=device)

    # k_scale is folded into the softmax scale (SGLang does the same).
    txda_decode_attention_fwd_sglang(
        q,
        k_buffer,
        v_buffer,
        o,
        fl_lse,
        kv_indptr,
        kv_indices,
        None,
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

    # Older flag_gems releases ship the _tsingmicro backend without this op; keep
    # SGLang's own decode_attention_fwd in that case.
    if load_txda_op("decode_attention_fwd_sglang") is None:
        _log.skipped("flag_gems has no decode_attention_fwd_sglang")
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
