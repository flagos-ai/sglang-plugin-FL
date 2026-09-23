# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Iluvatar CoreX.

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch


@lru_cache(maxsize=1)
def _packed_decode_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        out,
        state,
        indices,
        scale,
        stride_qkv: tl.constexpr,
        stride_a: tl.constexpr,
        stride_b: tl.constexpr,
        stride_state: tl.constexpr,
        stride_indices: tl.constexpr,
        H: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        L2NORM: tl.constexpr,
    ):
        block_v, batch_head = tl.program_id(0), tl.program_id(1)
        batch, value_head = batch_head // HV, batch_head % HV
        query_head = value_head // (HV // H)
        offsets_k = tl.arange(0, BK)
        offsets_v = block_v * BV + tl.arange(0, BV)
        mask_k = offsets_k < K
        mask_v = offsets_v < V
        mask_state = mask_v[:, None] & mask_k[None, :]

        slot = tl.load(indices + batch * stride_indices).to(tl.int64)
        out_ptr = out + (batch * HV + value_head) * V + offsets_v
        if slot < 0:
            tl.store(out_ptr, 0.0, mask=mask_v)
            return

        state_ptr = (
            state
            + slot * stride_state
            + value_head * V * K
            + offsets_v[:, None] * K
            + offsets_k[None, :]
        )
        current = tl.load(state_ptr, mask=mask_state, other=0.0).to(tl.float32)

        packed = mixed_qkv + batch * stride_qkv
        q = tl.load(
            packed + query_head * K + offsets_k, mask=mask_k, other=0.0
        ).to(tl.float32)
        k = tl.load(
            packed + H * K + query_head * K + offsets_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        value = tl.load(
            packed + 2 * H * K + value_head * V + offsets_v,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        if L2NORM:
            q *= tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6)
            k *= tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)
        q *= scale

        gate = tl.load(a + batch * stride_a + value_head).to(tl.float32)
        gate += tl.load(dt_bias + value_head).to(tl.float32)
        gate = tl.where(gate <= 20.0, tl.log(1.0 + tl.exp(gate)), gate)
        gate *= -tl.exp(tl.load(A_log + value_head).to(tl.float32))
        update = tl.sigmoid(
            tl.load(b + batch * stride_b + value_head).to(tl.float32)
        ).to(b.dtype.element_ty).to(tl.float32)

        current *= tl.exp(gate)
        value = (value - tl.sum(current * k[None, :], axis=1)) * update
        current += value[:, None] * k[None, :]
        result = tl.sum(current * q[None, :], axis=1)
        tl.store(out_ptr, result, mask=mask_v)
        tl.store(state_ptr, current, mask=mask_state)

    return kernel


def packed_decode_gdn(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    state: torch.Tensor,
    out: torch.Tensor,
    state_indices: torch.Tensor,
    use_qk_l2norm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    import triton

    batch = mixed_qkv.shape[0]
    value_heads, value_dim, key_dim = state.shape[-3:]
    query_dim = (mixed_qkv.shape[1] - value_heads * value_dim) // 2
    query_heads = query_dim // key_dim
    block_k = triton.next_power_of_2(key_dim)
    block_v = min(triton.next_power_of_2(value_dim), 32)
    grid = (triton.cdiv(value_dim, block_v), batch * value_heads)
    _packed_decode_kernel()[grid](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        out,
        state,
        state_indices,
        float(scale),
        stride_qkv=mixed_qkv.stride(0),
        stride_a=a.stride(0),
        stride_b=b.stride(0),
        stride_state=state.stride(0),
        stride_indices=state_indices.stride(0),
        H=query_heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        L2NORM=bool(use_qk_l2norm),
        num_warps=1,
        num_stages=3,
    )
    return out, state


@lru_cache(maxsize=1)
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def recurrent(
        q,
        k,
        v,
        g,
        beta,
        state,
        indices,
        cu,
        out,
        checkpoints,
        H: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        NVB: tl.constexpr,
        SCALE: tl.constexpr,
        L2NORM: tl.constexpr,
        SAVE_CHECKPOINTS: tl.constexpr,
        BV: tl.constexpr,
        BK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        seq = pid // (HV * NVB)
        rem = pid - seq * HV * NVB
        hv = rem // NVB
        vb = rem - hv * NVB
        vv = vb * BV + tl.arange(0, BV)
        kk = tl.arange(0, BK)
        mask_v = vv < V
        mask_k = kk < K
        slot = tl.load(indices + seq)
        active = slot >= 0
        state_offset = ((slot * HV + hv) * V + vv[:, None]) * K + kk[None, :]
        current = tl.load(
            state + state_offset,
            mask=active & mask_v[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        begin = tl.load(cu + seq)
        end = tl.load(cu + seq + 1)
        chunk_base = 0
        previous = 0
        while previous < seq:
            previous_begin = tl.load(cu + previous)
            previous_end = tl.load(cu + previous + 1)
            chunk_base += (previous_end - previous_begin + 63) // 64
            previous += 1
        ratio: tl.constexpr = HV // H
        q_head = hv // ratio
        token = begin
        while token < end:
            q_value = tl.load(
                q + (token * H + q_head) * K + kk, mask=mask_k, other=0.0
            ).to(tl.float32)
            k_value = tl.load(
                k + (token * H + q_head) * K + kk, mask=mask_k, other=0.0
            ).to(tl.float32)
            if L2NORM:
                q_value *= tl.rsqrt(tl.sum(q_value * q_value, axis=0) + 1e-6)
                k_value *= tl.rsqrt(tl.sum(k_value * k_value, axis=0) + 1e-6)
            decay = tl.load(g + token * HV + hv).to(tl.float32)
            update = tl.load(beta + token * HV + hv).to(tl.float32)
            value = tl.load(
                v + (token * HV + hv) * V + vv, mask=mask_v, other=0.0
            ).to(tl.float32)
            current *= tl.exp(decay)
            prediction = tl.sum(current * k_value[None, :], axis=1)
            current += ((value - prediction) * update)[:, None] * k_value[None, :]
            result = tl.sum(current * q_value[None, :], axis=1) * SCALE
            tl.store(
                out + (token * HV + hv) * V + vv,
                result,
                mask=active & mask_v,
            )
            if SAVE_CHECKPOINTS:
                local_token = token - begin + 1
                chunk = chunk_base + local_token // 64 - 1
                checkpoint_offset = (
                    (chunk * HV + hv) * V + vv[:, None]
                ) * K + kk[None, :]
                tl.store(
                    checkpoints + checkpoint_offset,
                    current,
                    mask=active
                    & (local_token % 64 == 0)
                    & mask_v[:, None]
                    & mask_k[None, :],
                )
            token += 1
        tl.store(
            state + state_offset,
            current,
            mask=active & mask_v[:, None] & mask_k[None, :],
        )

    return recurrent


def recurrent_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    use_qk_l2norm_in_kernel: bool,
    out: Optional[torch.Tensor] = None,
    return_checkpoints: bool = False,
):
    import triton

    if q.dim() == 4:
        q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)
    if g.dim() == 3:
        g, beta = g.squeeze(0), beta.squeeze(0)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    g, beta = g.contiguous(), beta.contiguous()
    tokens, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[-2:]
    if value_heads % heads:
        raise ValueError("GDN value heads must be divisible by query heads")
    if out is None:
        out = v.new_empty((tokens, value_heads, value_dim))
    else:
        out = out.view(tokens, value_heads, value_dim)
    block_v = 8
    value_blocks = triton.cdiv(value_dim, block_v)
    batch = cu_seqlens.numel() - 1
    if return_checkpoints:
        max_chunks = triton.cdiv(tokens, 64) + batch
        checkpoints = q.new_empty((1, max_chunks, value_heads, value_dim, key_dim))
    else:
        checkpoints = initial_state
    _kernel()[(batch * value_heads * value_blocks,)](
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        initial_state_indices,
        cu_seqlens,
        out,
        checkpoints,
        H=heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        NVB=value_blocks,
        SCALE=float(scale),
        L2NORM=bool(use_qk_l2norm_in_kernel),
        SAVE_CHECKPOINTS=return_checkpoints,
        BV=block_v,
        BK=triton.next_power_of_2(key_dim),
        num_warps=4,
        num_stages=1,
    )
    return (out, checkpoints) if return_checkpoints else out
