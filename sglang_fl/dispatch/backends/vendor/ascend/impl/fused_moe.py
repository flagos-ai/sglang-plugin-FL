# Ascend FusedMoE operator implementation.
# Ports UnquantizedFusedMoEMethod.forward_npu (sglang/srt/layers/quantization/unquant.py).

from __future__ import annotations

import torch


def fused_moe_ascend(
    obj,
    layer: torch.nn.Module,
    dispatch_output,
):
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    x = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output

    original_dtype = x.dtype
    num_tokens = x.shape[0]
    topk_weights = topk_weights.to(x.dtype)
    topk_ids = topk_ids.to(torch.int32)
    num_experts = layer.num_experts
    top_k = layer.top_k or topk_ids.shape[1]

    hidden_states, expanded_row_idx, expert_tokens, _ = (
        torch.ops.npu.npu_moe_init_routing_v2(
            x,
            topk_ids,
            active_num=num_tokens * top_k,
            expert_num=num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, num_experts],
            quant_mode=-1,
        )
    )
    expert_tokens = expert_tokens.to(torch.int64)

    w13_bias = [layer.w13_weight_bias] if obj.with_bias else None
    w2_bias = [layer.w2_weight_bias] if obj.with_bias else None

    # gmm1: gate_up_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[layer.w13_weight],
        bias=w13_bias,
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    # activation
    if obj.moe_runner_config.activation == "npu_swiglu_oai":
        from sgl_kernel_npu.activation.swiglu_oai import swiglu_oai

        hidden_states = swiglu_oai(layer, hidden_states)
    elif obj.moe_runner_config.activation == "silu":
        hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
    else:
        from sglang.srt.layers.activation import GeluAndMul

        hidden_states = GeluAndMul()(hidden_states)

    # gmm2: down_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[layer.w2_weight],
        bias=w2_bias,
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
        hidden_states,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
        drop_pad_mode=2,
    )

    return StandardCombineInput(hidden_states=final_hidden_states)
