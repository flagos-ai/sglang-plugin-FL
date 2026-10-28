# SPDX-License-Identifier: Apache-2.0
"""Keep 3-row MRoPE positions out of SGLang 0.5.18's 1-D fused kernels."""

from types import MethodType

from sglang.srt.model_executor.forward_context import get_attn_backend


def install_mrope_indexer(indexer):
    original = indexer.forward

    def forward(
        self, x, q_lora, positions, forward_batch, layer_id, return_indices=True
    ):
        if positions.ndim == 1:
            return original(
                x, q_lora, positions, forward_batch, layer_id, return_indices
            )
        if self.dsa_enable_prefill_cp or forward_batch.attn_cp_metadata is not None:
            raise ValueError(
                "HYV4 multimodal indexer currently supports TP/PP without context parallelism"
            )
        from sglang.kernels.ops.attention.dsa.triton_kernel import act_quant
        from sglang.srt.layers.attention.dsa.dsa_indexer import (
            _broadcast_indexer_topk_from_rank0,
        )
        from sglang.srt.state_capturer.indexer_topk import maybe_capture_indexer_topk

        metadata = get_attn_backend().get_indexer_metadata(layer_id, forward_batch)
        if metadata is None:
            return None
        query, key, raw = self._get_q_k_bf16(
            q_lora, x, positions, False, forward_batch=forward_batch
        )
        q, scale = act_quant(query, self.block_size, self.scale_fmt)
        self._store_index_k_cache(
            forward_batch=forward_batch, layer_id=layer_id, key=key, act_quant=act_quant
        )
        if not return_indices:
            return None
        if raw is None:
            raw = self._weights_proj_bf16_in_fp32_out(x)
        weights = (
            raw.float().unsqueeze(-1) * scale * self.softmax_scale * self.n_heads**-0.5
        )
        mode = forward_batch.forward_mode
        if (
            mode.is_decode_or_idle()
            or mode.is_target_verify()
            or mode.is_draft_extend_v2()
        ):
            result = self._get_topk_paged(forward_batch, layer_id, q, weights, metadata)
        else:
            result = self._get_topk_ragged(
                False, forward_batch, layer_id, q, weights, metadata
            )
        return maybe_capture_indexer_topk(
            layer_id, _broadcast_indexer_topk_from_rank0(result)
        )

    indexer.forward = MethodType(forward, indexer)
