# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Reference mhc_sinkhorn operator implementation using PyTorch.
#
# Independent eager reference for the fused Sinkhorn operator. Kept free of
# any Triton/dispatch code so numeric tests can compare against it without
# sharing implementation paths.

from __future__ import annotations

import torch


def mhc_sinkhorn_torch(comb: torch.Tensor, eps: float, repeat: int) -> torch.Tensor:
    """Apply `repeat` Sinkhorn normalization rounds to the last two dims.

    Mirrors the GLM mhc combinator reference: initial row softmax over the
    last dim, then alternating column/row divisions with `eps` in every
    denominator. Operates in float32 internally and writes the result back
    in the input dtype, matching the fused kernel numerics.

    Args:
        comb: [.., N, N] tensor.
        eps: additive epsilon, same placement as the fused kernel.
        repeat: total number of normalization rounds (>= 1).

    Returns:
        Normalized tensor with the input dtype.
    """
    if comb.dim() < 2:
        raise ValueError(f"comb must have at least 2 dims, got shape {tuple(comb.shape)}")
    if comb.shape[-2] != comb.shape[-1]:
        raise ValueError(f"comb must be square on the last two dims, got {tuple(comb.shape)}")
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")

    dtype = comb.dtype
    c = comb.float()
    c = c.softmax(-1) + eps
    c = c / (c.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        c = c / (c.sum(-1, keepdim=True) + eps)
        c = c / (c.sum(-2, keepdim=True) + eps)
    return c.to(dtype)
