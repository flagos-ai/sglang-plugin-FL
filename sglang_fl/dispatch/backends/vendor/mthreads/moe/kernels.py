# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Exact campaign kernel/launcher bodies, relocated without a core overwrite.

Quantized/TMA paths are not selected by this plugin's adapter. Their existing
branches are retained to avoid changing the compiled BF16 kernel during the
relocation; unchanged helpers remain owned by the pinned SGLang baseline.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch
import triton
import triton.language as tl
from sglang.srt.layers.moe.moe_runner.triton_utils import (
    fused_moe_triton_kernels as _core,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
    write_zeros_to_output,
    should_enable_swap_ab,
    fused_moe_kernel_gptq_awq,
    scaled_fp8_quant,
    per_token_group_quant_fp8,
    sglang_per_token_group_quant_fp8,
    per_token_quant_int8,
    per_token_group_quant_int8,
    sglang_per_token_group_quant_int8,
    _set_triton_tma_allocator,
    _get_b_tma_desc_cached,
)

_is_cuda = False
padding_size = _core.padding_size
TensorDescriptor = getattr(_core, "TensorDescriptor", None)


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    a_desc,
    b_ptr,
    b_desc,
    bias_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    add_mask_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_bias_e,
    stride_bias_n,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    even_Ks: tl.constexpr,
    c_sorted: tl.constexpr,
    filter_expert: tl.constexpr,
    swap_ab: tl.constexpr,
    FUSE_ADD_TO_OUTPUT: tl.constexpr,
    FUSE_SUM_ALL_REDUCE: tl.constexpr,
    ROUTER_TOPK: tl.constexpr,
    FUSE_SWIGLU: tl.constexpr = False,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.

    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    # FUSE_SWIGLU writes the I=N/2 activated columns directly.  The caller
    # correspondingly launches only the half-width output grid; the weight
    # tensor itself stays canonical [gate rows, up rows].
    num_pid_n = tl.cdiv(N // 2 if FUSE_SWIGLU else N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts_i32 = tl.load(expert_ids_ptr + pid_m)
    off_experts = off_experts_i32.to(tl.int64)

    if filter_expert and off_experts == -1:
        if FUSE_SWIGLU:
            # C is half-width in this mode. The down GEMM also skips filtered
            # expert blocks, so no zero fill is needed here.
            return
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        if not FUSE_ADD_TO_OUTPUT:
            # skip the zero-write to preserve existing values.
            write_zeros_to_output(
                c_ptr,
                stride_cm,
                stride_cn,
                pid_n,
                N,
                offs_token,
                token_mask,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                compute_type,
            )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if a_desc is not None:
        assert use_fp8_w8a8 and group_n > 0 and group_k > 0
        start_offs_m = pid_m * BLOCK_SIZE_M
    else:
        a_ptrs = a_ptr + (
            offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
        )

    if b_desc is not None:
        start_offs_n = pid_n * BLOCK_SIZE_N
    elif FUSE_SWIGLU:
        gate_b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        )
        up_b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] * stride_bk + (offs_bn[None, :] + N // 2) * stride_bn)
        )
    else:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        )

    if bias_ptr is not None:
        bias = tl.load(
            bias_ptr + off_experts * stride_bias_e + offs_bn[None, :] * stride_bias_n
        )
    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            if a_desc is not None:
                a_scale_ptrs = a_scale_ptr + offs_token_id * stride_asm
            else:
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            if BLOCK_SIZE_N > group_n:
                offs_bsn = offs_bn // group_n
            else:
                offs_bsn = pid_n * BLOCK_SIZE_N // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    if FUSE_SWIGLU:
        gate_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        up_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    elif swap_ab:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if a_desc is not None:
            a = a_desc.load([start_offs_m, k_start])
        elif even_Ks:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                other=0.0,
            )

        if FUSE_SWIGLU:
            # Canonical W13 is laid out as [gate, up].  Load both halves for
            # the same output tile and keep independent FP32 accumulators.
            if even_Ks:
                gate_b = tl.load(gate_b_ptrs)
                up_b = tl.load(up_b_ptrs)
            else:
                gate_b = tl.load(
                    gate_b_ptrs,
                    mask=offs_k[:, None] < K - k_start,
                    other=0.0,
                )
                up_b = tl.load(
                    up_b_ptrs,
                    mask=offs_k[:, None] < K - k_start,
                    other=0.0,
                )
        elif b_desc is not None:
            b = (
                b_desc.load([off_experts_i32, start_offs_n, k_start])
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
                .T
            )
        elif even_Ks:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)

        # We accumulate along the K dimension.
        if FUSE_SWIGLU:
            gate_accumulator += tl.dot(a, gate_b)
            up_accumulator += tl.dot(a, up_b)
        elif use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                if swap_ab:
                    a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))
                    a_scale, b_scale = b_scale, a_scale
                if BLOCK_SIZE_N > group_n:
                    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
                else:
                    accumulator += tl.dot(a, b) * (a_scale[:, None] * b_scale)
            else:
                if use_fp8_w8a8:
                    if swap_ab:
                        a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        if a_desc is None:
            a_ptrs += BLOCK_SIZE_K * stride_ak
        if b_desc is None and not FUSE_SWIGLU:
            b_ptrs += BLOCK_SIZE_K * stride_bk
        if FUSE_SWIGLU:
            gate_b_ptrs += BLOCK_SIZE_K * stride_bk
            up_b_ptrs += BLOCK_SIZE_K * stride_bk

    if FUSE_SWIGLU:
        # Round each GEMM result to BF16 before evaluating SwiGLU.  This
        # matches the unfused full-width BF16 buffer exactly while retaining
        # the canonical [gate, up] weight layout.
        gate_bf16 = gate_accumulator.to(compute_type)
        up_bf16 = up_accumulator.to(compute_type)
        gate_f = gate_bf16.to(tl.float32)
        up_f = up_bf16.to(tl.float32)
        out_act = (gate_f * tl.sigmoid(gate_f) * up_f).to(compute_type)
    else:
        if swap_ab:
            accumulator = tl.trans(accumulator, (1, 0))

        if use_int8_w8a16:
            accumulator *= b_scale
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k == 0 or group_n == 0:
                accumulator *= a_scale * b_scale

        if bias_ptr is not None:
            accumulator += bias

        if MUL_ROUTED_WEIGHT:
            moe_weight = tl.load(
                topk_weights_ptr + offs_token, mask=token_mask, other=0
            )
            accumulator *= moe_weight[:, None]

        accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if FUSE_SWIGLU:
        # The fused grid is half-width, so every program owns one contiguous
        # output tile of the activated intermediate.
        offs_half = offs_cn
        if c_sorted:
            c_ptrs = (
                c_ptr
                + stride_cm * offs_token_id[:, None]
                + stride_cn * offs_half[None, :]
            )
        else:
            c_ptrs = (
                c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_half[None, :]
            )
        c_mask = token_mask[:, None] & (offs_half[None, :] < N // 2)
        tl.store(c_ptrs, out_act, mask=c_mask)
    elif FUSE_ADD_TO_OUTPUT:
        # Accumulate into existing output with per-token mask.
        offs_token_out = offs_token // ROUTER_TOPK
        add_mask = tl.load(add_mask_ptr + offs_token_out, mask=token_mask, other=False)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & add_mask[:, None] & (offs_cn[None, :] < N)
        existing = tl.load(c_ptrs, mask=c_mask, other=0.0)
        tl.store(c_ptrs, existing + accumulator, mask=c_mask)
    elif FUSE_SUM_ALL_REDUCE:
        offs_token_out = offs_token // ROUTER_TOPK
        c_ptrs = (
            c_ptr + stride_cm * offs_token_out[:, None] + stride_cn * offs_cn[None, :]
        )
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask)
    else:
        if c_sorted:
            c_ptrs = (
                c_ptr
                + stride_cm * offs_token_id[:, None]
                + stride_cn * offs_cn[None, :]
            )
        else:
            c_ptrs = (
                c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
            )
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor],
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    B_zp: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
    a_use_tma: bool = False,
    b_use_tma: bool = False,
    c_sorted: bool = False,
    filter_expert: bool = True,
    fuse_sum_all_reduce: bool = False,
    router_topk: int = 1,
    fuse_add_to_output: bool = False,
    add_output_mask: Optional[torch.Tensor] = None,
    fuse_swiglu: bool = False,
) -> None:
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    if fuse_swiglu:
        assert not (use_fp8_w8a8 or use_int8_w8a8 or use_int8_w8a16 or use_int4_w4a16)
        assert bias is None
        assert not mul_routed_weight
        assert not (fuse_add_to_output or fuse_sum_all_reduce)
        assert compute_type == tl.bfloat16
        assert config["BLOCK_SIZE_N"] % 2 == 0

    if use_fp8_w8a8:
        swap_ab = should_enable_swap_ab(config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"])
    else:
        swap_ab = False

    padded_size = 0
    if use_fp8_w8a8:
        assert B_scale is not None
        if block_shape is None:
            # activation tensor-wise fp8 quantization, dynamic or static
            padded_size = padding_size
            # activations apply per-token quantization when weights apply per-channel quantization by default
            A, A_scale = scaled_fp8_quant(
                A, A_scale, use_per_token_if_dynamic=per_channel_quant
            )
        else:
            # activation block-wise fp8 quantization
            assert len(block_shape) == 2
            block_n, block_k = block_shape[0], block_shape[1]
            if _is_cuda:
                A, A_scale = sglang_per_token_group_quant_fp8(A, block_k)
            else:
                A, A_scale = per_token_group_quant_fp8(A, block_k)
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
            assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]
            assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]
    elif use_int8_w8a8:
        assert B_scale is not None
        if block_shape is None:
            # activation channel-wise int8 quantization
            assert (
                per_channel_quant
            ), "int8 quantization only supports channel-wise quantization except for block-wise quantization"
            A, A_scale = per_token_quant_int8(A)
        else:
            # activation block-wise int8 quantization
            assert len(block_shape) == 2
            block_n, block_k = block_shape[0], block_shape[1]
            if _is_cuda:
                A, A_scale = sglang_per_token_group_quant_int8(A, block_k)
            else:
                A, A_scale = per_token_group_quant_int8(A, block_k)
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
            assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]
            assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]
    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    output_n = B.shape[1] // 2 if fuse_swiglu else B.shape[1]
    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(output_n, META["BLOCK_SIZE_N"]),
    )

    K = B.shape[2] - padded_size
    if K % config["BLOCK_SIZE_K"] == 0:
        even_Ks = True
    else:
        even_Ks = False

    if fuse_sum_all_reduce:
        assert not c_sorted, "fuse_sum_all_reduce only supports c_sorted=False"
    if fuse_add_to_output:
        assert (
            not fuse_sum_all_reduce
        ), "fuse_add_to_output and fuse_sum_all_reduce are mutually exclusive"
        assert (
            add_output_mask is not None
        ), "add_output_mask required when fuse_add_to_output=True"

    if (
        (use_int8_w8a16 or use_int4_w4a16)
        and block_shape is not None
        and block_shape[1] > 0
    ):
        assert (
            not fuse_sum_all_reduce
        ), "fuse_sum_all_reduce is not supported for GPTQ/AWQ kernels"
        assert B_scale is not None and B_scale.ndim == 3
        assert B_zp is None or B_zp.ndim == 3
        assert bias is None
        fused_moe_kernel_gptq_awq[grid](
            A,
            B,
            C,
            B_scale,
            B_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            B.shape[1],
            A.shape[1],
            sorted_token_ids.shape[0],
            topk_ids.numel(),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(-2),
            C.stride(-1),
            B_scale.stride(0),
            B_scale.stride(2),
            B_scale.stride(1),
            B_zp.stride(0) if B_zp is not None else 0,
            B_zp.stride(2) if B_zp is not None else 0,
            B_zp.stride(1) if B_zp is not None else 0,
            group_size=block_shape[1],
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            has_zp=B_zp is not None,
            use_int4_w4a16=use_int4_w4a16,
            use_int8_w8a16=use_int8_w8a16,
            even_Ks=even_Ks,
            filter_expert=filter_expert,
            **config,
        )

    else:
        if a_use_tma or b_use_tma:
            _set_triton_tma_allocator()

        if a_use_tma:
            a_desc = TensorDescriptor(
                A, A.shape, A.stride(), [config["BLOCK_SIZE_M"], config["BLOCK_SIZE_K"]]
            )
        else:
            a_desc = None
        if b_use_tma:
            # B is constant weights -> cache descriptor
            b_desc = _get_b_tma_desc_cached(
                B,
                config["BLOCK_SIZE_N"],
                config["BLOCK_SIZE_K"],
            )
        else:
            b_desc = None

        fused_moe_kernel[grid](
            A,
            a_desc,
            B,
            b_desc,
            bias,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            add_output_mask,
            B.shape[1],
            B.shape[2] - padded_size,
            sorted_token_ids.shape[0],
            topk_ids.numel(),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            bias.stride(0) if bias is not None else 0,
            bias.stride(1) if bias is not None else 0,
            C.stride(-2),
            C.stride(-1),
            A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
            A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
            B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
            B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
            B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
            0 if block_shape is None else block_shape[0],
            0 if block_shape is None else block_shape[1],
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            per_channel_quant=per_channel_quant,
            even_Ks=even_Ks,
            c_sorted=c_sorted,
            filter_expert=filter_expert,
            swap_ab=swap_ab,
            FUSE_ADD_TO_OUTPUT=fuse_add_to_output,
            FUSE_SUM_ALL_REDUCE=fuse_sum_all_reduce,
            ROUTER_TOPK=router_topk,
            FUSE_SWIGLU=fuse_swiglu,
            **config,
        )
