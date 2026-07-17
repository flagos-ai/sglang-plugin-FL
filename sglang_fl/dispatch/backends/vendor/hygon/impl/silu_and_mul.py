# Bridge: SiluAndMul
#
# SGLang signature: forward_cuda(self, x: Tensor) -> Tensor
# Dispatch signature: fn(obj, x: Tensor) -> Tensor
# Mapping: trivial (1:1)

from __future__ import annotations

import torch



def silu_and_mul_hcu(obj, x: torch.Tensor) -> torch.Tensor:
    """SGLang SiluAndMul forward → dispatch call_op("silu_and_mul", ...)."""
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    from lightop import fuse_silu_and_mul
    fuse_silu_and_mul(x, out)
    return out