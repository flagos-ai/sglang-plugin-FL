# TXDA rotary embedding operator implementations.
#
# TODO: Replace NotImplementedError with torch_txda native kernel once verified
# on hardware. Current behavior: falls back to flaggems (Triton) or reference.

from __future__ import annotations

import torch


def rotary_embedding_txda(
    obj,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool = False,
    inplace: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError(
        "rotary_embedding_txda: no torch_txda kernel wired yet; falling back to flaggems/reference"
    )
