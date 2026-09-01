# TXDA GemmaRMSNorm operator implementation.
#
# TODO: Replace NotImplementedError with torch_txda native kernel once verified
# on hardware. Current behavior: falls back to reference.

from __future__ import annotations

from typing import Optional, Union

import torch


def gemma_rms_norm_txda(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    raise NotImplementedError(
        "gemma_rms_norm_txda: no torch_txda kernel wired yet; falling back to flaggems/reference"
    )
