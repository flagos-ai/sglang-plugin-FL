# Iluvatar vendor GemmaRMSNorm — delegates to SGLang's native kernel.

from __future__ import annotations

from typing import Optional, Union

import torch


def gemma_rms_norm_iluvatar(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    GemmaRMSNorm using SGLang's native kernel.
    """
    return obj._forward_impl(x, residual)
