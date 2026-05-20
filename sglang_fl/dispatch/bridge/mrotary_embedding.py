# Bridge: MRotaryEmbedding
#
# SGLang signature:
#   forward_cuda(self, positions, query, key, fused_set_kv_buffer_arg=None)
#     -> tuple[Tensor, Tensor]
#
# Dispatch signatures:
#   mrotary_embedding:
#     fn(obj, positions, query, key) -> tuple[Tensor, Tensor]
#
#   mrotary_embedding_with_kv_cache:
#     fn(obj, positions, query, key, fused_set_kv_buffer_arg)
#     -> tuple[Tensor, Tensor]
#
# SGLang-specific handling:
#   - positions can be 1D [num_tokens] or 2D [3, num_tokens] (multimodal)
#   - mrope_section splits rotary_dim across 3 axes (text/image/video)
#   - fused_set_kv_buffer_arg: dispatch as separate fused op
#     "mrotary_embedding_with_kv_cache" (fused RoPE + KV cache write)

from __future__ import annotations

from typing import Tuple

import torch

from sglang_fl.dispatch import call_op


def mrotary_embedding_bridge(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    fused_set_kv_buffer_arg=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SGLang MRotaryEmbedding forward → dispatch call_op("mrotary_embedding", ...)."""
    if fused_set_kv_buffer_arg is not None:
        return call_op(
            "mrotary_embedding_with_kv_cache",
            self,
            positions,
            query,
            key,
            fused_set_kv_buffer_arg,
        )
    return call_op("mrotary_embedding", self, positions, query, key)
