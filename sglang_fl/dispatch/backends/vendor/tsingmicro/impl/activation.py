# TXDA activation operator implementations.
#
# TODO: Replace NotImplementedError with torch_txda native kernel once verified
# on hardware. Current behavior: falls back to reference.

from __future__ import annotations

import torch


def silu_and_mul_txda(obj, x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError(
        "silu_and_mul_txda: no torch_txda kernel wired yet; falling back to flaggems/reference"
    )