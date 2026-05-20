# MUSA MRotaryEmbedding operator implementation.
#
# TODO: Replace NotImplementedError with torch_musa native kernel once verified
# on hardware. Current behavior: falls back to reference.

from __future__ import annotations

from typing import Tuple

import torch


def mrotary_embedding_musa(
    obj,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-modal rotary position embedding on MUSA.

    Args:
        obj: The MRotaryEmbedding instance
        positions: Position tensor (1D or 2D for multi-modal)
        query: Query tensor
        key: Key tensor

    Returns:
        Tuple of (embedded_query, embedded_key)
    """
    raise NotImplementedError(
        "mrotary_embedding_musa: no torch_musa kernel wired yet; falling back to flaggems/reference"
    )
