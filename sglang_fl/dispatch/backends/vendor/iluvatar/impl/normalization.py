from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_iluvatar(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    from .triton_ops import rms_norm

    out = rms_norm(x, obj.weight, obj.variance_epsilon, residual)
    if out is not None:
        return out

    if residual is not None:
        x = x + residual
        residual = x
    output = torch.nn.functional.rms_norm(
        x.float(), (x.shape[-1],), obj.weight.float(), obj.variance_epsilon
    ).to(x.dtype)
    return (output, residual) if residual is not None else output
