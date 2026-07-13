# MUSA normalization operator implementations.
#
# RMSNorm:      mirrors sglang RMSNorm.forward_musa() (layernorm.py:341)
#               — nn.functional.rms_norm (no residual, pure torch_musa ATen op)
#               — sgl_kernel.fused_add_rmsnorm (residual, MUSA-compiled kernel;
#                 imported at build time: if _is_musa: from sgl_kernel import ...)
#
# GemmaRMSNorm: mirrors sglang GemmaRMSNorm._forward_impl() (layernorm.py:578)
#               reached via base-class forward_musa → forward_cuda → _forward_impl.
#               — sgl_kernel.gemma_rmsnorm (no residual)
#               — sgl_kernel.gemma_fused_add_rmsnorm (residual)
#               gemma_rmsnorm internally applies norm(x) * (1 + weight);
#               pass obj.weight.data (the raw trained weight, not +1 shifted).

from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_musa(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    if not x.is_contiguous():
        x = x.contiguous()
    if residual is not None:
        from sgl_kernel import fused_add_rmsnorm
        fused_add_rmsnorm(x, residual, obj.weight.data, obj.variance_epsilon)
        return x, residual
    return torch.nn.functional.rms_norm(
        x, (x.shape[-1],), obj.weight.data, obj.variance_epsilon
    )


def gemma_rms_norm_musa(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    from sgl_kernel import gemma_fused_add_rmsnorm, gemma_rmsnorm
    if residual is not None:
        gemma_fused_add_rmsnorm(x, residual, obj.weight.data, obj.variance_epsilon)
        return x, residual
    return gemma_rmsnorm(x, obj.weight.data, obj.variance_epsilon)
