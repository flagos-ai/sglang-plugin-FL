# MetaX activation implementations.

from __future__ import annotations

import torch


def silu_and_mul_metax(obj, x: torch.Tensor) -> torch.Tensor:
    from sglang_fl.dispatch.backends.vendor.cuda.impl.activation import (
        silu_and_mul_cuda,
    )

    return silu_and_mul_cuda(obj, x)
