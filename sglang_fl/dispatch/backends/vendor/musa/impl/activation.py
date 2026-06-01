# MUSA activation operator implementations.
#
# TODO: Replace NotImplementedError with torch_musa native kernel once verified
# on hardware. Current behavior: falls back to reference.

from __future__ import annotations

import torch


def silu_and_mul_musa(obj, x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError(
        "mrotary_embedding_musa: no torch_musa kernel wired yet; falling back to flaggems/reference"
    )
