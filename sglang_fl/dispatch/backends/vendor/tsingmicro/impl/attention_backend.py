# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TXDA attention backend for SGLang.

``TxdaAttnBackend`` subclasses SGLang's ``TritonAttnBackend`` and routes both
attention kernels to the flaggems ``_tsingmicro`` backend:

* prefill/extend -> ``extend_attention_fwd``
* decode         -> ``decode_attention_fwd_sglang``

Metadata, CUDA-graph handling and the deterministic unified extend kernel stay
inherited.

The kernels are bound in ``__init__`` (``self.extend_attention_fwd`` /
``self.decode_attention_fwd``) rather than monkey-patched onto SGLang's
modules, so the routing is scoped to this backend and does not depend on
SGLang's import order.

Two behavioural differences from the stock triton backend:

1. ``forward_extend`` does **not** call ``k.contiguous()`` / ``v.contiguous()``:
   the flaggems kernel reads through the real tensor strides, so the ``split`` +
   ``view`` strided q/k/v produced by the model are passed straight through,
   avoiding large copies on TXDA.
2. Decode uses the flaggems two-segment CSR kernel instead of SGLang's
   token-by-token gather.  Over a ``[pool_size, kv_heads, head_dim]`` pool with
   ``kv_indptr`` / ``kv_indices``, the kernel walks each request's leading run of
   *physically contiguous* slots affinely and hands the rest to a masked gather.
   That was measured ~23x faster at the 2k decode scenario (84 ms vs 1.9 s per
   layer).

``prefix_lens`` (the length of each request's contiguous run) is derived inside
the flaggems op by ``_compute_prefix_lens`` / ``_prefix_len_kernel`` when
``prefix_lens=None`` is passed -- a native TXDA scan, no host sync.  Computing it
in Python instead used to go through flaggems' *generic* ``cumprod`` (no
``_tsingmicro`` specialization), which lowers to ``tts.get_structured_state``
and faults the device (HPGR error 0x600000F).  The decode kernel is therefore
only installed when the installed flaggems op supports the in-kernel scan.

Over-estimating ``prefix_lens`` is silent: the kernel would read physically
contiguous slots belonging to another request and return plausible but wrong
logits, with no crash.  Keep the merge step of that scan off flaggems' generic
``min_dim`` -- see ``_TXDA_EXCLUDED_ATEN_OPS`` in ``sglang_fl/__init__.py``.

When the installed flag_gems predates either op (see
:mod:`...tsingmicro.flaggems_compat`), or a decode call uses a feature the op
does not model, the kernel stays SGLang's own.

Decode routing is gated by ``SGLANG_FL_DECODE_ATTN_FLAGOS`` (default ``1`` =
enabled; set ``0`` to disable).

Note: speculative decoding's ``TritonMultiStepDraftBackend`` builds plain
``TritonAttnBackend`` instances for the draft steps, so those keep SGLang's
triton decode kernel.
"""

from __future__ import annotations

import inspect
import logging
import os
import sys

import torch

from sglang.srt.layers.attention.triton_backend import (
    TritonAttnBackend,
    logit_capping_mod,
)

from sglang_fl.dispatch.backends.vendor.tsingmicro.flaggems_compat import (
    load_txda_op,
)

logger = logging.getLogger(__name__)

_DECODE_ENABLED_KEY = "SGLANG_FL_DECODE_ATTN_FLAGOS"
_DECODE_OP_NAME = "decode_attention_fwd_sglang"

# One-shot per-process diagnostics (decode attention runs per layer, per step).
_warned_decode_fallback = False


def _decode_attention_enabled() -> bool:
    """Whether to route decode attention to the flaggems kernel (default on)."""
    return os.getenv(_DECODE_ENABLED_KEY, "1").lower() in ("1", "true", "yes", "on")


def _supports_in_kernel_prefix(fn) -> bool:
    """Whether ``fn`` accepts ``prefix_lens=None`` and derives it on device.

    Older flag_gems releases shipped ``decode_attention_fwd_sglang`` with a
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


def _make_flaggems_decode_fwd(fl_op, orig_fwd):
    """Build a ``decode_attention_fwd``-compatible callable backed by ``fl_op``.

    ``orig_fwd`` is SGLang's kernel as bound by ``TritonAttnBackend.__init__``;
    it stays the fallback whenever the flaggems op does not model the call.
    """

    def _decode_attention_fwd(
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
        global _warned_decode_fallback

        # Features the flaggems op does not model -> SGLang's kernel.
        if (
            sinks is not None
            or xai_temperature_len > 0
            or v_scale != 1.0
            or k_buffer.dtype != q.dtype
            or v_buffer.dtype != q.dtype
        ):
            if not _warned_decode_fallback:
                _warned_decode_fallback = True
                logger.warning(
                    "decode_attention FALLBACK: unsupported feature "
                    "(sinks=%s xai=%s v_scale=%s k_dtype=%s v_dtype=%s); "
                    "using SGLang's triton kernel",
                    sinks is not None,
                    xai_temperature_len,
                    v_scale,
                    k_buffer.dtype,
                    v_buffer.dtype,
                )
            return orig_fwd(
                q, k_buffer, v_buffer, o, kv_indptr, kv_indices,
                attn_logits, attn_lse, num_kv_splits, max_kv_splits,
                sm_scale, k_scale, v_scale,
                logit_cap=logit_cap,
                sinks=sinks,
                xai_temperature_len=xai_temperature_len,
            )

        batch = q.shape[0]
        head_num = q.shape[1]

        # attn_logits/attn_lse (SGLang's split-K scratch) are unused: the
        # flaggems op runs its own stage-1/stage-2 split internally and writes
        # straight to ``o``.  Its lse output is not consumed by SGLang, so a
        # throwaway buffer is enough.
        fl_lse = torch.empty(batch, head_num, dtype=torch.float32, device=q.device)

        # k_scale is folded into the softmax scale (SGLang does the same).
        fl_op(
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

    return _decode_attention_fwd


class TxdaAttnBackend(TritonAttnBackend):
    def __init__(self, model_runner, **kwargs):
        super().__init__(model_runner, **kwargs)

        # Replace SGLang's extend kernel with the flaggems _tsingmicro
        # implementation.  The unified deterministic kernel remains SGLang's.
        #
        # Older flag_gems releases do not ship this op; ``load_txda_op`` returns
        # None then and we keep the kernel ``super().__init__`` installed.
        txda_extend_attention_fwd = load_txda_op("extend_attention_fwd")
        self._use_flaggems_extend = txda_extend_attention_fwd is not None
        if self._use_flaggems_extend:
            self.extend_attention_fwd = txda_extend_attention_fwd

        # Same for decode: ``super().__init__`` bound SGLang's triton kernel to
        # ``self.decode_attention_fwd``; swap in the flaggems CSR two-segment
        # kernel, keeping the original as the per-call fallback.
        self._use_flaggems_decode = False
        txda_decode_attention_fwd = load_txda_op(_DECODE_OP_NAME)
        if not _decode_attention_enabled():
            logger.info(
                "decode attention: %s=0, using SGLang's triton kernel",
                _DECODE_ENABLED_KEY,
            )
        elif txda_decode_attention_fwd is None:
            logger.info(
                "decode attention: flag_gems has no %s, using SGLang's kernel",
                _DECODE_OP_NAME,
            )
        elif not _supports_in_kernel_prefix(txda_decode_attention_fwd):
            logger.info(
                "decode attention: flag_gems %s cannot derive prefix_lens "
                "in-kernel, using SGLang's kernel",
                _DECODE_OP_NAME,
            )
        else:
            self._use_flaggems_decode = True
            # ``disable`` matches how ``super().__init__`` wrapped the original:
            # decode must stay out of torch.compile tracing.
            self.decode_attention_fwd = torch.compiler.disable(
                _make_flaggems_decode_fwd(
                    txda_decode_attention_fwd, self.decode_attention_fwd
                )
            )
            logger.info(
                "decode attention -> flag_gems two-segment kernel "
                "(prefix_lens computed in-kernel)"
            )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        save_kv_cache: bool = True,
        sinks=None,
    ):
        # No flag_gems extend kernel available (older flag_gems): SGLang's own
        # forward_extend is identical except that it copies k/v with
        # .contiguous() first, which the flag_gems kernel does not need.
        if not self._use_flaggems_extend:
            return super().forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, sinks
            )

        from sglang.srt.layers.radix_attention import AttentionType
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if k is None and v is None:
            pool = forward_batch.token_to_kv_pool
            cache_loc = forward_batch.out_cache_loc
            if isinstance(pool, SWAKVPool) and pool.layers_mapping[layer.layer_id][1]:
                cache_loc = pool.translate_loc_from_full_to_swa(cache_loc)
            k_buffer, v_buffer = pool.get_kv_buffer(layer.layer_id)
            k = k_buffer[cache_loc]
            v = v_buffer[cache_loc]
        elif k is None or v is None:
            raise ValueError("Both k and v should be None or not None")
        else:
            # Save KV cache first (must do this before the kernel)
            if save_kv_cache:
                if self.use_mla or layer.k_scale is None:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer,
                        forward_batch.out_cache_loc,
                        k,
                        v,
                    )
                else:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer,
                        forward_batch.out_cache_loc,
                        k.clone(),
                        v.clone(),
                        layer.k_scale,
                        layer.v_scale,
                    )

        logits_soft_cap = logit_capping_mod(
            layer.logit_capping_method, layer.logit_cap
        )

        causal = True
        if (
            layer.is_cross_attention
            or layer.attn_type == AttentionType.ENCODER_ONLY
            or (
                layer.attn_type == AttentionType.DECODER_BIDIRECTIONAL
                and self.allow_bidirectional_attention_in_extend
            )
        ):
            causal = False

        # Deterministic mode: use the inherited unified 1-stage kernel.
        if self.enable_deterministic:
            return self._forward_extend_unified(
                q, o, layer, forward_batch, causal, logits_soft_cap, sinks
            )

        # Normal mode: two-stage kernel (prefix paged KV + extend triangle).
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            sliding_window_size = layer.sliding_window_size
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
            window_kv_offsets = self.forward_metadata.window_kv_offsets
        else:
            sliding_window_size = -1
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
            window_kv_offsets = None

        if layer.k_scale is not None and layer.v_scale is not None:
            k_descale = layer.k_scale_float
            v_descale = layer.v_scale_float
        else:
            k_descale = 1.0
            v_descale = 1.0

        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k,
            v,
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            self.forward_metadata.qo_indptr,
            kv_indptr,
            kv_indices,
            self.forward_metadata.custom_mask,
            causal,
            self.forward_metadata.mask_indptr,
            self.forward_metadata.max_extend_len,
            k_descale,
            v_descale,
            layer.scaling,
            logit_cap=logits_soft_cap,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_kv_offsets=window_kv_offsets,
            xai_temperature_len=layer.xai_temperature_len,
        )
        return o
