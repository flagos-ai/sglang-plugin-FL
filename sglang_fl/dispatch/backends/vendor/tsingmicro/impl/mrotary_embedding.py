# TXDA MRotaryEmbedding operator implementation.
#
# TODO: Replace NotImplementedError with torch_txda native kernel once verified
# on hardware. Current behavior: falls back to reference.

from __future__ import annotations

from typing import Tuple

import torch


def mrotary_embedding_txda(
    obj,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError(
        "mrotary_embedding_txda: no torch_txda kernel wired yet; falling back to flaggems/reference"
    )
