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

# FlagOS mhc_sinkhorn operator implementation.
#
# Fused Sinkhorn normalization for GLM hyper-connection (mhc) combinator
# matrices: `repeat` alternating row/column normalizations on the last two
# dims of a [s, n, n] tensor, fused into a single Triton launch. The eager
# equivalent issues ~2 kernels per iteration (2 * repeat launches); this
# fused path targets eager-mode prefill where those launches are real work.
#
# NOTE on graph mode: under a captured decode graph the eager chain has no
# per-kernel launch overhead, so this fused kernel is expected to be
# performance-neutral or slower there. It is registered but NOT routed by
# default (no op_backends entry for "mhc_sinkhorn"); enable it per platform
# via the dispatch config when measured beneficial.

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sinkhorn_kernel(
    comb_ptr,  # [..., N, N], any float dtype, modified in place
    eps,
    REPEAT: tl.constexpr,
    N: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    sidx = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    ni = tl.arange(0, N)
    off = sidx[:, None, None] * (N * N) + ni[None, :, None] * N + ni[None, None, :]
    c = tl.load(comb_ptr + off).to(tl.float32)

    m = tl.max(c, axis=2)
    e = tl.exp(c - m[:, :, None])
    c = e / tl.sum(e, axis=2)[:, :, None]
    c = c + eps
    c = c / (tl.sum(c, axis=1)[:, None, :] + eps)
    for _ in tl.static_range(REPEAT - 1):
        c = c / (tl.sum(c, axis=2)[:, :, None] + eps)
        c = c / (tl.sum(c, axis=1)[:, None, :] + eps)
    tl.store(comb_ptr + off, c.to(comb_ptr.dtype.element_ty))


def mhc_sinkhorn_flagos(comb: torch.Tensor, eps: float, repeat: int) -> torch.Tensor:
    """Apply `repeat` Sinkhorn normalization rounds to the last two dims.

    Args:
        comb: [.., N, N] tensor (any float dtype); must be contiguous.
            Modified in place and also returned, mirroring the eager
            reference contract.
        eps: additive epsilon applied after the initial row softmax and in
            every denominator, matching the reference numerics.
        repeat: total number of normalization rounds (>= 1).

    Returns:
        The same tensor object with normalized values.
    """
    if comb.dim() < 2:
        raise ValueError(f"comb must have at least 2 dims, got shape {tuple(comb.shape)}")
    n = comb.shape[-1]
    if comb.shape[-2] != n:
        raise ValueError(f"comb must be square on the last two dims, got {tuple(comb.shape)}")
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")
    if not comb.is_contiguous():
        comb = comb.contiguous()
    s = comb.numel() // (n * n)
    if s == 0:
        return comb
    block_s = min(triton.next_power_of_2(s), 128)
    grid = (triton.cdiv(s, block_s),)
    _sinkhorn_kernel[grid](comb, eps, REPEAT=repeat, N=n, BLOCK_S=block_s)
    return comb
