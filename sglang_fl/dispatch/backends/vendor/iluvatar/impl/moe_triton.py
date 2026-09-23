# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Iluvatar CoreX.

from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def first(
        x,
        w13,
        ids,
        mid,
        TOPK: tl.constexpr,
        H: tl.constexpr,
        I: tl.constexpr,
        TRANSPOSED: tl.constexpr,
        BH: tl.constexpr,
        BI: tl.constexpr,
    ):
        row, block = tl.program_id(0), tl.program_id(1)
        ii = block * BI + tl.arange(0, BI)
        hh = tl.arange(0, BH)
        mask_i = ii < I
        expert = tl.load(ids + row)
        values = tl.load(
            x + (row // TOPK) * H + hh, mask=hh < H, other=0.0
        ).to(tl.float32)
        if TRANSPOSED:
            gate_ptr = w13 + (expert * H + hh[None, :]) * (2 * I) + ii[:, None]
            up_ptr = gate_ptr + I
        else:
            gate_ptr = w13 + (expert * 2 * I + ii[:, None]) * H + hh[None, :]
            up_ptr = gate_ptr + I * H
        weight_mask = mask_i[:, None] & (hh[None, :] < H)
        gate_weight = tl.load(gate_ptr, mask=weight_mask, other=0.0).to(tl.float32)
        up_weight = tl.load(up_ptr, mask=weight_mask, other=0.0).to(tl.float32)
        gate = tl.sum(gate_weight * values[None, :], axis=1)
        up = tl.sum(up_weight * values[None, :], axis=1)
        tl.store(mid + row * I + ii, gate * tl.sigmoid(gate) * up, mask=mask_i)

    @triton.jit
    def second(
        mid,
        w2,
        ids,
        routed,
        I: tl.constexpr,
        H: tl.constexpr,
        TRANSPOSED: tl.constexpr,
        BI: tl.constexpr,
        BH: tl.constexpr,
    ):
        row, block = tl.program_id(0), tl.program_id(1)
        hh = block * BH + tl.arange(0, BH)
        ii = tl.arange(0, BI)
        mask_h = hh < H
        expert = tl.load(ids + row)
        values = tl.load(mid + row * I + ii, mask=ii < I, other=0.0).to(
            tl.float32
        )
        if TRANSPOSED:
            weight_ptr = w2 + (expert * I + ii[None, :]) * H + hh[:, None]
        else:
            weight_ptr = w2 + (expert * H + hh[:, None]) * I + ii[None, :]
        weight = tl.load(
            weight_ptr,
            mask=mask_h[:, None] & (ii[None, :] < I),
            other=0.0,
        ).to(tl.float32)
        tl.store(routed + row * H + hh, tl.sum(weight * values[None, :], axis=1), mask=mask_h)

    @triton.jit
    def reduce(
        routed,
        weights,
        out,
        TOPK: tl.constexpr,
        H: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        token, block = tl.program_id(0), tl.program_id(1)
        hh = block * BLOCK + tl.arange(0, BLOCK)
        mask = hh < H
        acc = tl.zeros((BLOCK,), tl.float32)
        for index in tl.static_range(TOPK):
            value = tl.load(
                routed + (token * TOPK + index) * H + hh,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc += value * tl.load(weights + token * TOPK + index).to(tl.float32)
        tl.store(out + token * H + hh, acc, mask=mask)

    return first, second, reduce


def fused_experts(
    hidden: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    import triton

    tokens, hidden_size = hidden.shape
    topk = topk_ids.shape[1]
    canonical = (
        w13.ndim == 3
        and w2.ndim == 3
        and w13.shape[1] == 2 * w2.shape[2]
        and w13.shape[2] == hidden_size
        and w2.shape[1] == hidden_size
    )
    transposed = (
        w13.ndim == 3
        and w2.ndim == 3
        and w13.shape[1] == hidden_size
        and w13.shape[2] == 2 * w2.shape[1]
        and w2.shape[2] == hidden_size
    )
    if not canonical and not transposed:
        raise ValueError(
            f"unsupported MoE weight layout: hidden={tuple(hidden.shape)}, "
            f"w13={tuple(w13.shape)}, w2={tuple(w2.shape)}"
        )
    intermediate = w2.shape[1] if transposed else w2.shape[2]
    rows = tokens * topk
    ids = topk_ids.to(torch.int32).contiguous().view(-1)
    mid = hidden.new_empty((rows, intermediate))
    routed = hidden.new_empty((rows, hidden_size))
    out = hidden.new_empty((tokens, hidden_size))
    first, second, reduce = _kernels()
    first[(rows, triton.cdiv(intermediate, 4))](
        hidden,
        w13,
        ids,
        mid,
        TOPK=topk,
        H=hidden_size,
        I=intermediate,
        TRANSPOSED=transposed,
        BH=triton.next_power_of_2(hidden_size),
        BI=4,
        num_warps=8,
        num_stages=1,
    )
    second[(rows, triton.cdiv(hidden_size, 16))](
        mid,
        w2,
        ids,
        routed,
        I=intermediate,
        H=hidden_size,
        TRANSPOSED=transposed,
        BI=triton.next_power_of_2(intermediate),
        BH=16,
        num_warps=4,
        num_stages=1,
    )
    reduce[(tokens, triton.cdiv(hidden_size, 256))](
        routed,
        topk_weights,
        out,
        TOPK=topk,
        H=hidden_size,
        BLOCK=256,
        num_warps=4,
        num_stages=1,
    )
    return out
