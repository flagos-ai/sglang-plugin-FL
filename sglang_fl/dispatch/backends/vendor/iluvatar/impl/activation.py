from __future__ import annotations

import torch
import torch.nn.functional as F


def silu_and_mul_iluvatar(obj, x: torch.Tensor) -> torch.Tensor:
    del obj
    from .triton_ops import silu_and_mul

    out = silu_and_mul(x)
    if out is not None:
        return out
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up
