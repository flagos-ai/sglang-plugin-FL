# Bridge: MRotaryEmbedding
#
# SGLang signature:
#   forward_cuda(self, positions, query, key, fused_set_kv_buffer_arg=None)
#     -> tuple[Tensor, Tensor]
#
# Dispatch signature:
#   fn(obj, positions, query, key) -> tuple[Tensor, Tensor]
#
# SGLang-specific handling:
#   - positions can be 1D [num_tokens] or 2D [3, num_tokens] (multimodal)
#   - mrope_section splits rotary_dim across 3 axes (text/image/video)
#   - fused_set_kv_buffer_arg: not supported by dispatch, fall through to native

from __future__ import annotations

from typing import Tuple

import torch



def mrotary_embedding_hcu(
    obj,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SGLang MRotaryEmbedding forward → dispatch call_op("mrotary_embedding", ...)."""
    # fused_set_kv_buffer_arg requires native kernel — fall through
    if positions.ndim == 2 and hasattr(obj, "mrope_section") and obj.mrope_section:
        return obj.forward_triton(positions, query, key)
    # 1D positions: use standard sgl_kernel rope
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

    return RotaryEmbedding.forward_cuda(obj, positions, query, key)
