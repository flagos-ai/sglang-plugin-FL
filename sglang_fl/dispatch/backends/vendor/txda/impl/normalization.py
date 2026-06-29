# TXDA normalization operator implementations.
#
# RMSNorm:      uses torch.nn.functional.rms_norm (pure torch ATen op
#               dispatched through torch_txda)
#               For residual case, uses torch-native add + rms_norm.
#
# GemmaRMSNorm: uses torch.nn.functional.rms_norm + weight scaling.
#               gemma_rmsnorm(x) = rms_norm(x) * (1 + weight)

from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_txda(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    if not x.is_contiguous():
        x = x.contiguous()
    if residual is not None:
        x.add_(residual)
        residual.copy_(x)
    out = torch.nn.functional.rms_norm(
        x, (x.shape[-1],), obj.weight.data, obj.variance_epsilon
    )
    if residual is not None:
        return out, residual
    return out


def gemma_rms_norm_txda(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    if not x.is_contiguous():
        x = x.contiguous()
    if residual is not None:
        x.add_(residual)
        residual.copy_(x)
    out = torch.nn.functional.rms_norm(
        x, (x.shape[-1],), obj.weight.data, obj.variance_epsilon
    )
    out.mul_(1.0 + obj.weight.data)
    if residual is not None:
        return out, residual
    return out
