# GCU normalization operator implementations.

from __future__ import annotations

from typing import Optional, Union

import torch
from sgl_kernel import rmsnorm, fused_add_rmsnorm
from sgl_kernel import gemma_fused_add_rmsnorm, gemma_rmsnorm

def rms_norm_gcu(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    if x.numel() == 0:
        if residual is not None:
            return x, residual
        return x

    if residual is not None:
        fused_add_rmsnorm(x, residual, obj.weight.data, obj.variance_epsilon)
        return x, residual
    out = rmsnorm(x, obj.weight.data, obj.variance_epsilon)
    return out


def gemma_rms_norm_gcu(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    
    if residual is not None:
        gemma_fused_add_rmsnorm(x, residual, obj.weight.data, obj.variance_epsilon)
        return x, residual
    out = gemma_rmsnorm(x, obj.weight.data, obj.variance_epsilon)
    return out