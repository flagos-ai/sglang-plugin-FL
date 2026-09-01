# GCU activation operator implementations.

from __future__ import annotations
import torch

def silu_and_mul_gcu(obj, x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication using GCU.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    from sgl_kernel import silu_and_mul
    return silu_and_mul(x)

