# GCU rotary embedding operator implementations.

from __future__ import annotations

import torch

def rotary_embedding_gcu(
    obj,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool = False,
    inplace: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
        Apply rotary position embedding on enflame.
    """

    raise NotImplementedError("rotary_embedding: GCU kernel not integrated yet; falling back to flaggems/reference implementation.")
