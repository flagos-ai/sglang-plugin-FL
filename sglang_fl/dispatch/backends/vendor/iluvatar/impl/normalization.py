from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_iluvatar(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    variance_size = getattr(obj, "variance_size_override", None)
    if variance_size is not None:
        if x.shape[-1] < variance_size:
            raise ValueError(
                f"Expected hidden_size to be at least {variance_size}, "
                f"but found: {x.shape[-1]}"
            )
        if residual is not None:
            x = x + residual
            residual = x
        x_float = x.float()
        variance = x_float[..., :variance_size].pow(2).mean(-1, keepdim=True)
        output = (
            x_float
            * torch.rsqrt(variance + obj.variance_epsilon)
            * obj.weight.float()
        ).to(x.dtype)
        return (output, residual) if residual is not None else output

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
