# Bridge: RMSNorm
#
# SGLang signature:
#   forward_cuda(self, x, residual=None, post_residual_addition=None)
#     -> Tensor | tuple[Tensor, Tensor]
#
# Dispatch signature:
#   fn(obj, x, residual=None) -> Tensor | tuple[Tensor, Tensor]
#
# SGLang-specific handling:
#   - post_residual_addition: added to residual before passing to dispatch

from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_hcu(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    RMS normalization using sgl_kernel.

    Args:
        obj: The calling obj (provides obj.weight, obj.variance_epsilon)
        x: Input tensor
        residual: Optional residual tensor

    Returns:
        Normalized tensor, or tuple of (normalized, residual) if residual provided
    """
    from sgl_kernel import rmsnorm as sgl_rms_norm
    from sgl_kernel import fused_add_rmsnorm as sgl_fused_add_rms_norm

    weight = obj.weight
    epsilon = obj.variance_epsilon

    if residual is not None:
        sgl_fused_add_rms_norm(x, residual, weight, epsilon)
        return x, residual
    else:
        out = torch.empty_like(x)
        sgl_rms_norm(out, x, weight, epsilon)
        return out
