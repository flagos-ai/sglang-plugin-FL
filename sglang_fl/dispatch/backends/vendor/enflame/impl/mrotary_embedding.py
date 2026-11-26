# GCU MRotaryEmbedding operator implementation.

from __future__ import annotations

from typing import Tuple

import torch
from sgl_kernel import apply_rope_with_cos_sin_cache_inplace, mrotary_embedding

def mrotary_embedding_gcu(
    obj,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    if positions.ndim == 1:
        # Text-only 1D positions are int64 from the scheduler. GCU native
        # kernels index positions as int32, so cast before the kernel, same as
        # the 2D branch below and the other enflame patches.
        positions = positions.to(dtype=torch.int32).contiguous()
        apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=obj.head_size,
            cos_sin_cache=obj.cos_sin_cache,
            is_neox=obj.is_neox_style,
        )
    elif positions.ndim == 2:
        # GCU native kernels index positions as int32; mrope_positions arrives
        # as (3, seq_len) int64 from the scheduler. Cast before the kernel,
        # matching the other enflame patches (multimodal_mask, qwen3_vl,
        # req_to_token).
        positions = positions.to(dtype=torch.int32).contiguous()
        mrotary_embedding(
            positions=positions,
            query=query,
            key=key,
            head_size=obj.head_size,
            cos_sin_cache=obj.cos_sin_cache,
            is_neox=obj.is_neox_style,
            mrope_section=obj.mrope_section,
            mrope_interleaved=obj.mrope_interleaved,
        )

    return query, key
