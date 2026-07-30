# Kunlunxin activation operator implementations using xflashinfer ops.

from __future__ import annotations

import torch


def silu_and_mul_kunlunxin(obj, x: torch.Tensor) -> torch.Tensor:
    """SiLU-and-multiply on Kunlunxin via the xflashinfer ``swiglu_ep`` op.

    Args:
        obj: The calling layer (unused; kept for interface consistency).
        x: Input tensor of shape ``[..., 2*d]``.

    Returns:
        Output tensor of shape ``[..., d]``.

    ``xtorch_ops.swiglu_ep`` wraps ``torch.ops.xflashinfer_ops.silu_and_mul_ep``
    and takes 2-D ``(N, 2*d)`` input plus a preallocated ``(N, d)`` output, so
    the tensors are flattened into views before the call. This keeps the fused
    activation off sglang's ``jit_kernel`` c++20 nvcc path, which fails on
    Kunlun's CUDA 11.7 host toolchain.
    """
    import xtorch_ops

    d = x.shape[-1] // 2
    out = x.new_empty(*x.shape[:-1], d)
    xtorch_ops.swiglu_ep(x.view(-1, x.shape[-1]), out.view(-1, d))
    return out
