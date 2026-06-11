# FlagOS MRotaryEmbedding operator implementation.

from __future__ import annotations

from typing import Tuple

import torch


def mrotary_embedding_flagos(
    obj,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-level MRotaryEmbedding entry used by sglang-plugin-FL.

    Mirrors sglang's MRotaryEmbedding.forward_cuda dispatch:
      - 2D positions with mrope_section -> triton_mrope_fused
      - otherwise                       -> 1D RoPE

    Note: use attribute access (obj.xxx) instead of obj.__dict__[xxx]
    because cos_sin_cache is an nn.Module registered buffer stored in
    obj._buffers, not in obj.__dict__.
    """
    from flaggems_sglang import triton_mrope_fused, _rope_1d

    mrope_section = getattr(obj, "mrope_section", None)

    if positions.ndim == 2 and mrope_section:
        triton_mrope_fused(
            query, key, obj.cos_sin_cache, positions, mrope_section,
            obj.head_size, obj.rotary_dim,
            getattr(obj, "mrope_interleaved", False),
            getattr(obj, "mrope_interleaved_glm", False),
            obj.is_neox_style, getattr(obj, "axis_map", None),
        )
    else:
        _rope_1d(
            query, key, obj.cos_sin_cache, positions,
            obj.head_size, obj.rotary_dim, obj.is_neox_style,
        )

    return query, key
