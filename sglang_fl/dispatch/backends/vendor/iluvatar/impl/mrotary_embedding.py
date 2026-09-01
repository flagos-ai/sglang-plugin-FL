# Iluvatar vendor MRotaryEmbedding.

from __future__ import annotations

from typing import Tuple

import torch


def mrotary_embedding_iluvatar(
    obj,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MRotaryEmbedding via SGLang."""
    if positions.ndim == 2 and hasattr(obj, "mrope_section") and obj.mrope_section:
        return obj.forward_triton(positions, query, key)
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

    return RotaryEmbedding.forward_cuda(obj, positions, query, key)
