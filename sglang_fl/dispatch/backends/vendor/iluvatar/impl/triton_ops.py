# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Iluvatar CoreX.

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch

_MAX_FUSED_SIZE = 65536


def _eligible(x: torch.Tensor) -> bool:
    return x.is_cuda and x.dtype in (torch.float16, torch.bfloat16)


@lru_cache(maxsize=1)
def _silu_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(x, out, half: tl.constexpr, size, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < size
        row = offsets // half
        col = offsets - row * half
        gate = tl.load(x + row * (2 * half) + col, mask=mask).to(tl.float32)
        up = tl.load(x + row * (2 * half) + half + col, mask=mask).to(tl.float32)
        tl.store(out + offsets, gate * tl.sigmoid(gate) * up, mask=mask)

    return kernel


def silu_and_mul(x: torch.Tensor) -> Optional[torch.Tensor]:
    if (
        not _eligible(x)
        or not x.is_contiguous()
        or x.ndim == 0
        or x.shape[-1] % 2
    ):
        return None

    import triton

    half = x.shape[-1] // 2
    size = x.numel() // 2
    out = x.new_empty((*x.shape[:-1], half))
    _silu_kernel()[(triton.cdiv(size, 256),)](
        x, out, half, size, BLOCK=256, num_warps=4, num_stages=1
    )
    return out


@lru_cache(maxsize=1)
def _rotary_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        x,
        cos,
        sin,
        positions,
        heads: tl.constexpr,
        head_size: tl.constexpr,
        half: tl.constexpr,
        NEOX: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        program = tl.program_id(0)
        token = program // heads
        head = program - token * heads
        dim = tl.arange(0, BLOCK)
        mask = dim < head_size
        rotate = mask & (dim < 2 * half)
        if NEOX:
            cache_dim = dim % half
            partner_dim = tl.where(dim < half, dim + half, dim - half)
            sign = tl.where(dim < half, -1.0, 1.0)
        else:
            cache_dim = dim // 2
            partner_dim = tl.where(dim % 2 == 0, dim + 1, dim - 1)
            sign = tl.where(dim % 2 == 0, -1.0, 1.0)
        position = tl.load(positions + token)
        cache_offset = position * half + cache_dim
        scale_cos = tl.load(cos + cache_offset, mask=rotate, other=0.0)
        scale_sin = tl.load(sin + cache_offset, mask=rotate, other=0.0)
        base = (token * heads + head) * head_size
        value = tl.load(x + base + dim, mask=mask, other=0.0).to(tl.float32)
        partner = tl.load(
            x + base + partner_dim, mask=rotate, other=0.0
        ).to(tl.float32)
        tl.store(
            x + base + dim,
            tl.where(
                rotate,
                value * scale_cos + sign * partner * scale_sin,
                value,
            ),
            mask=mask,
        )

    return kernel


def rotary_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    interleaved: bool,
) -> bool:
    if (
        not _eligible(query)
        or not _eligible(key)
        or not query.is_contiguous()
        or not key.is_contiguous()
        or query.dtype != key.dtype
        or query.shape[0] != key.shape[0]
        or query.shape[0] != positions.numel()
        or cos.device != query.device
        or sin.device != query.device
        or not cos.is_contiguous()
        or not sin.is_contiguous()
    ):
        return False

    import triton

    positions = positions.reshape(-1).contiguous()
    head_size = query.shape[-1]
    half = cos.shape[-1]
    if 2 * half > head_size or key.shape[-1] != head_size:
        return False
    block = triton.next_power_of_2(head_size)
    kernel = _rotary_kernel()
    for tensor in (query, key):
        heads = tensor.numel() // (positions.numel() * head_size)
        kernel[(positions.numel() * heads,)](
            tensor,
            cos,
            sin,
            positions,
            heads=heads,
            head_size=head_size,
            half=half,
            NEOX=not interleaved,
            BLOCK=block,
            num_warps=4,
            num_stages=1,
        )
    return True


@lru_cache(maxsize=1)
def _rms_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        x,
        weight,
        out,
        residual,
        residual_out,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        WEIGHT_OFFSET: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < hidden
        offsets = row * hidden + cols
        values = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            values += tl.load(residual + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            tl.store(residual_out + offsets, values, mask=mask)
            # sglang normalizes the stored residual, so read back the rounding.
            values = tl.load(residual_out + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
        variance = tl.sum(values * values, axis=0) / hidden
        values *= tl.rsqrt(variance + eps)
        scale = tl.load(weight + cols, mask=mask, other=0.0).to(tl.float32)
        scale += WEIGHT_OFFSET
        tl.store(out + offsets, values * scale, mask=mask)

    return kernel


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: Optional[torch.Tensor] = None,
    weight_offset: float = 0.0,
):
    if (
        not _eligible(x)
        or not x.is_contiguous()
        or x.ndim == 0
        or not weight.is_contiguous()
        or weight.device != x.device
        or weight.dtype != x.dtype
        or weight.numel() != x.shape[-1]
    ):
        return None
    if residual is not None and (
        residual.shape != x.shape
        or residual.device != x.device
        or residual.dtype != x.dtype
        or not residual.is_contiguous()
    ):
        return None

    import triton

    hidden = x.shape[-1]
    block = triton.next_power_of_2(hidden)
    if block > _MAX_FUSED_SIZE:
        return None
    rows = x.numel() // hidden
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x) if residual is not None else out
    residual_arg = residual if residual is not None else x
    _rms_kernel()[(rows,)](
        x,
        weight,
        out,
        residual_arg,
        residual_out,
        hidden,
        eps,
        HAS_RESIDUAL=residual is not None,
        WEIGHT_OFFSET=float(weight_offset),
        BLOCK=block,
        num_warps=4 if block <= 4096 else 8,
        num_stages=1,
    )
    return (out, residual_out) if residual is not None else out
