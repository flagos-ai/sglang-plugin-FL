# SPDX-License-Identifier: Apache-2.0
"""HYV4 attention sinks on the SGLang 0.5.18 DSA layout."""

import torch
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

from .attention_policy import freeze_role_policy


class HYV4SparseAttnBackend(DeepseekSparseAttnBackend):
    def __init__(self, model_runner, **kwargs):
        self._hy4_mla_role = freeze_role_policy(model_runner)
        self._hy4_forward_mode = None
        args = model_runner.server_args
        for name in ("dsa_prefill_backend", "dsa_decode_backend"):
            if getattr(args, name) != "flashmla_sparse":
                raise ValueError(
                    f"HYV4 attention sinks require --{name.replace('_', '-')} flashmla_sparse"
                )
        super().__init__(model_runner, **kwargs)
        self._hy4_sink = None

    def forward_extend(self, q, k, v, layer, forward_batch, *args, **kwargs):
        self._hy4_sink = layer.hy4_attention_sink
        self._hy4_forward_mode = getattr(forward_batch, "forward_mode", None)
        try:
            return super().forward_extend(
                q, k, v, layer, forward_batch, *args, **kwargs
            )
        finally:
            self._hy4_sink = None
            self._hy4_forward_mode = None

    def forward_decode(self, q, k, v, layer, forward_batch, *args, **kwargs):
        self._hy4_sink = layer.hy4_attention_sink
        self._hy4_forward_mode = getattr(forward_batch, "forward_mode", None)
        try:
            return super().forward_decode(
                q, k, v, layer, forward_batch, *args, **kwargs
            )
        finally:
            self._hy4_sink = None
            self._hy4_forward_mode = None

    def _forward_flashmla_sparse(
        self, q_all, kv_cache, v_head_dim, page_table_1, sm_scale, topk_length=None
    ):
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        tokens, heads, head_dim = q_all.shape
        required_heads = 128 if self.device_sm_major >= 10 else 64
        sink = self._hy4_sink
        if sink is None or sink.shape != (heads,) or sink.dtype != torch.float32:
            raise ValueError("HYV4 requires one FP32 attention sink per local head")
        if heads % required_heads:
            if required_heads % heads:
                raise ValueError("HYV4 head shard cannot be padded for FlashMLA")
            query = q_all.new_zeros((tokens, required_heads, head_dim))
            query[:, :heads] = q_all
            padded_sink = sink.new_full((required_heads,), -float("inf"))
            padded_sink[:heads] = sink
        else:
            query, padded_sink = (q_all, sink)
        if topk_length is not None and topk_length.shape[0] != tokens:
            topk_length = None
        from .sparse_mla_active_split import maybe_run

        candidate = maybe_run(
            self,
            query,
            kv_cache,
            page_table_1.unsqueeze(1),
            sm_scale,
            padded_sink,
            topk_length,
            heads,
            v_head_dim,
        )
        if candidate is not None:
            return candidate
        output, _, _ = flash_mla_sparse_fwd(
            q=query,
            kv=kv_cache,
            indices=page_table_1.unsqueeze(1),
            sm_scale=sm_scale,
            d_v=v_head_dim,
            topk_length=topk_length,
            attn_sink=padded_sink,
        )
        return output[:, :heads]
