from __future__ import annotations

from typing import Tuple

import torch


def mrotary_embedding_iluvatar(
    obj,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if (
        obj.cos_sin_cache.device != query.device
        or obj.cos_sin_cache.dtype != query.dtype
    ):
        obj.cos_sin_cache = obj.cos_sin_cache.to(query.device, dtype=query.dtype)

    if positions.ndim == 2 and obj.mrope_section:
        from .mrotary_triton import mrotary_embedding

        return mrotary_embedding(obj, positions, query, key)

    if positions.ndim == 1:
        from .triton_ops import rotary_embedding

        cache_cos, cache_sin = obj.cos_sin_cache.chunk(2, dim=-1)
        query_view = query.view(query.shape[0], -1, obj.head_size)
        key_view = key.view(key.shape[0], -1, obj.head_size)
        if rotary_embedding(
            query_view,
            key_view,
            cache_cos,
            cache_sin,
            positions,
            not obj.is_neox_style,
        ):
            return query, key

    from sglang_fl.dispatch.backends.reference.impl.mrotary_embedding import (
        mrotary_embedding_torch,
    )

    return mrotary_embedding_torch(obj, positions, query, key)
