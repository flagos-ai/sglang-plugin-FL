# Iluvatar normalization operator implementations using SGLang layer forward.

from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_iluvatar(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    RMS normalization via SGLang RMSNorm.forward_cuda.

    Args:
        obj: The calling RMSNorm instance (provides obj.weight, obj.variance_epsilon)
        x: Input tensor
        residual: Optional residual tensor (post_residual_addition merged by bridge)

    Returns:
        Normalized tensor, or tuple of (normalized, residual) if residual provided
    """
    return obj.forward_cuda(x, residual)
