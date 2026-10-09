# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TXDA attention backend for SGLang.

``TxdaAttnBackend`` subclasses SGLang's ``TritonAttnBackend`` and routes the
prefill/extend path to the flaggems ``_tsingmicro`` ``extend_attention_fwd``
Triton kernel.  Everything else (decode, metadata, CUDA-graph handling) is
inherited unchanged.

The one behavioural difference from the stock triton backend is that
``forward_extend`` does **not** call ``k.contiguous()`` / ``v.contiguous()``:
the flaggems kernel reads through the real tensor strides, so the ``split`` +
``view`` strided q/k/v produced by the model are passed straight through,
avoiding large copies on TXDA.

When the installed flag_gems predates ``extend_attention_fwd`` (see
:mod:`...tsingmicro.flaggems_compat`), the kernel stays SGLang's own and
``forward_extend`` simply delegates to ``TritonAttnBackend``.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.attention.triton_backend import (
    TritonAttnBackend,
    logit_capping_mod,
)

from sglang_fl.dispatch.backends.vendor.tsingmicro.flaggems_compat import (
    load_txda_op,
)


class TxdaAttnBackend(TritonAttnBackend):
    def __init__(self, model_runner, **kwargs):
        super().__init__(model_runner, **kwargs)

        # Replace SGLang's extend kernel with the flaggems _tsingmicro
        # implementation.  Decode and the unified deterministic kernel remain
        # SGLang's, as they are outside this task's scope.
        #
        # Older flag_gems releases do not ship this op; ``load_txda_op`` returns
        # None then and we keep the kernel ``super().__init__`` installed.
        txda_extend_attention_fwd = load_txda_op("extend_attention_fwd")
        self._use_flaggems_extend = txda_extend_attention_fwd is not None
        if self._use_flaggems_extend:
            self.extend_attention_fwd = txda_extend_attention_fwd

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
