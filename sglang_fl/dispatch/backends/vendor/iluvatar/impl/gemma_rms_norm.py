from __future__ import annotations

from typing import Optional, Union

import torch


def gemma_rms_norm_iluvatar(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    from .triton_ops import rms_norm

    weight = obj.weight
    out = rms_norm(
        x, weight, obj.variance_epsilon, residual, weight_offset=1.0
    )
    if out is not None:
        return out

    if residual is not None:
        x = x + residual
        residual = x
    values = x.float()
    values *= torch.rsqrt(values.square().mean(-1, keepdim=True) + obj.variance_epsilon)
    output = (values * (1 + weight.float())).to(x.dtype)
    return (output, residual) if residual is not None else output
