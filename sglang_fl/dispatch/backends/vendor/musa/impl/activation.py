# MUSA activation operator implementations.
#
# silu_and_mul: mirrors sglang's forward_musa from
#   python/sglang/multimodal_gen/runtime/layers/activation.py
#   which uses nn.SwishGLU — pure torch_musa ops, no Triton required.

from __future__ import annotations

import torch


def silu_and_mul_musa(obj, x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.nn.functional.silu(x[..., :d]) * x[..., d:]
