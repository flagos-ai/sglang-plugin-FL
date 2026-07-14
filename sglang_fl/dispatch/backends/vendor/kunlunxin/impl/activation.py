# Kunlunxin activation operator implementations using sgl_kernel.

from __future__ import annotations

import torch


def silu_and_mul_kunlunxin(obj, x: torch.Tensor) -> torch.Tensor:
    """SiLU-and-multiply on Kunlunxin via the klx ``sgl_kernel`` build.

    Args:
        obj: The calling layer (unused; kept for interface consistency).
        x: Input tensor of shape ``[..., 2*d]``.

    Returns:
        Output tensor of shape ``[..., d]``.

    NOTE on argument order: the Kunlun (klx) ``sgl_kernel`` build takes
    ``silu_and_mul(input, out)`` — the SAME order sglang itself uses on its
    ``_is_xpu`` branch (``silu_and_mul(x, out)``), NOT the ``(out, input)`` order
    of the upstream NVIDIA ``sgl_kernel`` build. This routes the fused activation
    onto the klx kernel and avoids sglang's ``jit_kernel`` c++20 nvcc path (which
    fails on Kunlun's CUDA 11.7 host toolchain).
    """
    import sgl_kernel  # noqa: F401

    d = x.shape[-1] // 2
    out = x.new_empty(*x.shape[:-1], d)
    torch.ops.sgl_kernel.silu_and_mul.default(out, x)
    return out
