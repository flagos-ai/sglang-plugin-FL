# Iluvatar activation operator implementations using SGLang layer forward.

from __future__ import annotations

import torch


def silu_and_mul_iluvatar(obj, x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication via SGLang.

    Args:
        obj: The calling SiluAndMul instance
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    return obj.forward_cuda(x)
