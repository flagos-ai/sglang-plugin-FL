# GCU FusedMoE operator implementation.

from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Any
import functools
import torch
import torch.nn.functional as F
import torch_gcu

from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

# max_block_m,swiglu_with_alpha_and_limit,
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    _use_aiter,_down_moe_use_tma,_is_cuda,_is_hip,moe_sum_reduce_torch_compile)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import moe_sum_reduce_triton
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import get_default_config
import sgl_kernel as sgl_ops

def moe_align_block_size_gcu(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor = None,
    topk_ids_size=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    # sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    # expert_ids.fill_(0)
    num_tokens_post_pad = torch.empty(
        (1), dtype=torch.int32, device=topk_ids.device
    )

    token_cnts_buffer = None
    cumsum_buffer = None
    if topk_ids_size is not None:
        torch.ops.sgl_kernel.moe_align_block_size_pad(
            topk_ids,
            topk_ids_size,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            token_cnts_buffer,
            cumsum_buffer,
        )
    else:
        torch.ops.sgl_kernel.moe_align_block_size(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            token_cnts_buffer,
            cumsum_buffer,
        )

    if expert_map is not None:
        expert_ids = torch_gcu.gcu.efficient.gcu_index(expert_map, [expert_ids])
    return sorted_ids, expert_ids, num_tokens_post_pad

def invoke_fused_moe_kernel_gcu(
    A: torch.Tensor,
    B: torch.Tensor,
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
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    block_shape: Optional[List[int]] = None,
    real_token_num=None,
    per_channel_quant: bool = False,
    bias: Optional[torch.Tensor]=None,
    A_scale_rec=None,
) -> None:
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8 or use_int8_w8a8:
        assert B_scale is not None
    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape and block_shape[0] == 0
        assert B_zp is None or B_zp.ndim == 3
    else:
        assert A_scale is None
        assert B_scale is None

    block_size = config["BLOCK_SIZE_M"]

    if use_fp8_w8a8 or use_int8_w8a16 or use_int4_w4a16 or use_int8_w8a8:
        if use_fp8_w8a8:
            B_zp = None
            if B.dtype != torch.int8:
                if block_shape is None:
                    group_size = -1
                else:
                    group_size = block_shape[1]
        elif use_int8_w8a8:
            B_zp = None
            group_size = 1
            if B_scale is not None and B_scale.ndim == 3:
                B_scale = B_scale.transpose(1,2).contiguous()
        elif use_int4_w4a16:
            A_scale = None
            group_size = block_shape[1]
        elif use_int8_w8a16:
            A_scale = None
            group_size = -1
            if B_scale.dim() == 3 and B_scale.shape[1] > 1:
                group_size = B.shape[2] // B_scale.shape[1]
                assert group_size in [64, 128], \
                    f"Unsupported shape, B: {B.shape}, B_scale: {B_scale.shape} for w8a16(int8)."
        else:
            raise NotImplementedError

        if use_fp8_w8a8 and B.dtype == torch.int8:
            # w4a8-fp8
            assert A_scale_rec is not None
            torch.ops.sgl_kernel.fused_moe_quant_kernel_ex(
                C,
                A,
                B,
                A_scale_rec,
                B_scale,
                B_zp,
                None,
                topk_weights,
                topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                real_token_num,
                mul_routed_weight,
                top_k,
                block_size,
                128,
                -1,
            )
        elif use_fp8_w8a8 and B.dtype == torch.float8_e4m3fn and block_shape is None:
            torch.ops.sgl_kernel.fused_moe_quant_kernel_ex(
                C,
                A,
                B,
                A_scale_rec,
                B_scale,
                B_zp,
                bias,
                topk_weights,
                topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                real_token_num,
                mul_routed_weight,
                top_k,
                block_size,
                -1,
                -1,
            )
        else:
            torch.ops.sgl_kernel.fused_moe_quant_kernel(
                C,
                A,
                B,
                A_scale,
                B_scale,
                group_size,
                B_zp,
                topk_weights,
                topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                mul_routed_weight,
                top_k,
                block_size,
                None,
                real_token_num,
            )
    else:
        topk_weights = topk_weights.to(torch.float32)  # WA for grouped_topk
        torch.ops.sgl_kernel.fused_moe_kernel(
            C,
            A,
            B,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            block_size,
            bias,
        )

def fused_experts_impl_gcu(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    inplace: bool = False,
    activation: str = "silu",
    is_gated: bool = True,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    filter_expert: bool = True,
):
    if not (use_fp8_w8a8 or use_int8_w8a8) or block_shape is not None or _use_aiter:
        padded_size = 0

    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.shape[1] // 2 == w1.shape[2], "Hidden size mismatch"
    else:
        assert (
            hidden_states.shape[1] == w1.shape[2] - padded_size
        ), f"Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = 64 * 1024
    M = min(num_tokens, CHUNK_SIZE)

    get_config_func = functools.partial(
        get_default_config,
        E=w2.shape[0],  # num_experts
        N=w2.shape[2],  # hidden_size
        K=w1.shape[2],  # intermediate_size
        topk=topk_ids.shape[1],
        dtype="",
        is_marlin=False,
        block_shape=block_shape,
    )

    config = get_config_func(M)

    down_config = None
    down_moe_use_tma = (
        _down_moe_use_tma()
        and down_config is not None
        and down_config.pop("USE_TMA", False)
    )
    topk = topk_ids.shape[1]
    max_padded_tokens = (
        min(M * topk, E + 1) * (max_block_m - 1) if down_moe_use_tma else 0
    )
    total_tokens = M * topk + max_padded_tokens
    cache = torch.empty(
        total_tokens * max(N, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = cache[: M * topk * w2.shape[1]].view(
        (M, topk, w2.shape[1]),
    )

    if no_combine:
        assert not inplace
        out_hidden_states = torch.empty(
            (num_tokens, topk, w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    elif inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            config, (down_config, _) = get_config_func(tokens_in_chunk)
            down_moe_use_tma = (
                _down_moe_use_tma()
                and down_config is not None
                and down_config.pop("USE_TMA", False)
            )
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]

        config["BLOCK_SIZE_M"] = 64 # hard code on gcu
        padded_tokens = (
            min(tokens_in_chunk * topk, E + 1) * (config["BLOCK_SIZE_M"] - 1)
            if down_moe_use_tma
            else 0
        )
        total_tokens = tokens_in_chunk * topk + padded_tokens
        intermediate_cache1 = cache[: total_tokens * N].view(
            (tokens_in_chunk, topk, N),
        )
        intermediate_cache2 = torch.empty(
            (total_tokens, N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size_gcu(
            curr_topk_ids, config["BLOCK_SIZE_M"], E
        )

        if use_fp8_w8a8:
            if block_shape is None:
                curr_hidden_states, a1_scale = sgl_ops.scaled_fp8_quant(curr_hidden_states, a1_scale)
            else:
                assert len(block_shape) == 2
                _, block_k = block_shape[0], block_shape[1]
                if a1_scale is None:
                    curr_hidden_states, a1_scale = sgl_ops.per_token_group_quant_fp8(curr_hidden_states, block_k)

        invoke_fused_moe_kernel_gcu(
            curr_hidden_states,
            w1,
            intermediate_cache1,
            a1_scale,
            w1_scale,
            w1_zp,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            topk_ids.shape[1],
            config,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            block_shape=block_shape,
            real_token_num= None,
            bias=b1
        )

        # Activation function with multiplication
        if activation == "silu" and is_gated:
            x = intermediate_cache1.view(-1, N)
            d = x.shape[-1] // 2
            intermediate_cache2.copy_(F.silu(x[..., :d]) * x[..., d:])
        elif activation == "gelu" and is_gated:
            assert gemm1_alpha is None, "gemm1_alpha is not supported for gelu"
            assert gemm1_limit is None, "gemm1_limit is not supported for gelu"
            if _is_cuda or _is_hip:
                x = intermediate_cache1.view(-1, N)
                d = x.shape[-1] // 2
                intermediate_cache2.copy_(F.gelu(x[..., :d]) * x[..., d:])
        # Activation function without multiplication
        elif activation == "silu" and not is_gated:
            intermediate_cache2 = F.silu(intermediate_cache1.view(-1, N))
        elif activation == "gelu" and not is_gated:
            intermediate_cache2 = F.gelu(intermediate_cache1.view(-1, N))
        elif activation == "relu2" and not is_gated:
            intermediate_cache2 = torch.square(F.relu(intermediate_cache1.view(-1, N)))
        else:
            raise ValueError(f"Unsupported activation: {activation=}, with {is_gated=}")

        if use_fp8_w8a8:
            if block_shape is None:
                intermediate_cache2, a2_scale = sgl_ops.scaled_fp8_quant(intermediate_cache2, a1_scale)
            else:
                assert len(block_shape) == 2
                _, block_k = block_shape[0], block_shape[1]
                if a2_scale is None:
                    intermediate_cache2, a2_scale = sgl_ops.per_token_group_quant_fp8(intermediate_cache2, block_k)
    
        invoke_fused_moe_kernel_gcu(
            intermediate_cache2,
            w2,
            (
                intermediate_cache3
                if not no_combine and topk_ids.shape[1] != 1
                else out_hidden_states[begin_chunk_idx:end_chunk_idx].unsqueeze(0)
            ),
            a2_scale,
            w2_scale,
            w2_zp,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            down_config or config,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            block_shape=block_shape,
            real_token_num=None,
            bias=b2,
        )

        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        if no_combine:
            pass
        elif _is_cuda:
            if topk_ids.shape[1] == 1 and routed_scaling_factor == 1.0:
                pass  # we write directly into out_hidden_states
            elif topk_ids.shape[1] == 2 and routed_scaling_factor == 1.0:
                torch.add(
                    intermediate_cache3[:, 0],
                    intermediate_cache3[:, 1],
                    out=out_hidden_states[begin_chunk_idx:end_chunk_idx],
                ).squeeze(dim=1)
            else:
                valid_in_chunk = torch.full(
                    (1,),
                    tokens_in_chunk,
                    dtype=torch.int32,
                    device=curr_hidden_states.device,
                )
                torch.ops.sgl_kernel.moe_sum_pad(
                    out_hidden_states[begin_chunk_idx:end_chunk_idx],
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    valid_in_chunk,
                    1,
                    False,
                )

        elif _is_hip:
            if _use_aiter:
                moe_sum( # noqa
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx],
                )
            else:
                # According to micro benchmark results, torch.compile can get better performance for small token.
                if tokens_in_chunk <= 32:
                    moe_sum_reduce_torch_compile(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )
                else:
                    moe_sum_reduce_triton(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )
        else:
            moe_sum_reduce_triton(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states[begin_chunk_idx:end_chunk_idx],
                routed_scaling_factor,
            )

    return out_hidden_states

def fused_experts_gcu(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_output: StandardTopKOutput,
    moe_runner_config: MoeRunnerConfig,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
):
    topk_weights, topk_ids, _ = topk_output
    filter_expert = (
        moe_runner_config.num_experts is None
        or moe_runner_config.num_experts != moe_runner_config.num_local_experts
    )

    assert not moe_runner_config.no_combine, "no combine + inplace makes no sense"
    fused_experts_impl_gcu(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        b1,
        b2,
        True,
        moe_runner_config.activation,
        moe_runner_config.is_gated,
        moe_runner_config.apply_router_weight_on_input,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        moe_runner_config.routed_scaling_factor,
        moe_runner_config.gemm1_alpha,
        moe_runner_config.gemm1_clamp_limit,
        filter_expert,
    )
    return hidden_states

def fused_moe_gcu(
    obj,
    layer: torch.nn.Module,
    dispatch_output,
):
    quant_info = TritonMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                b13=getattr(layer, "w13_weight_bias", None),
                b2=getattr(layer, "w2_weight_bias", None),
            )
    
    output = fused_experts_gcu(
        hidden_states=dispatch_output.hidden_states,
        w1=quant_info.w13_weight,
        w2=quant_info.w2_weight,
        topk_output=dispatch_output.topk_output,
        moe_runner_config=obj.runner.config,
        b1=quant_info.b13,
        b2=quant_info.b2,
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        per_channel_quant=quant_info.per_channel_quant,
        w1_scale=quant_info.w13_scale,
        w2_scale=quant_info.w2_scale,
        w1_zp=quant_info.w13_zp,
        w2_zp=quant_info.w2_zp,
        a1_scale=quant_info.a13_scale,
        a2_scale=quant_info.a2_scale,
        block_shape=quant_info.block_shape,
    )

    return StandardCombineInput(
        hidden_states=output,
    )
    