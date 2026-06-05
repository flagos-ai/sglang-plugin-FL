# FlagOS FusedMoE operator implementation.

from __future__ import annotations

import torch


def fused_moe_flagos(
    obj,
    layer: torch.nn.Module,
    dispatch_output,
):
    from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
    from sglang.srt.layers.moe.moe_runner.triton_kernels import TritonKernelsRunnerOutput
    from ..gems_sglang.fused_moe_kernel import (
        triton_kernel_fused_experts,
        triton_kernel_fused_experts_with_bias,
    )

    quant_info = TritonMoeQuantInfo(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        b13=getattr(layer, "w13_weight_bias", None),
        b2=getattr(layer, "w2_weight_bias", None),
    )
    
    hidden_states = dispatch_output.hidden_states

    common_kwargs = dict(
        routing_data=dispatch_output.routing_data,
        gather_indx=dispatch_output.gather_indx,
        scatter_indx=None if obj.runner.config.no_combine else dispatch_output.scatter_indx,
        inplace=False,
        activation=obj.runner.config.activation,
        apply_router_weight_on_input=obj.runner.config.apply_router_weight_on_input,
        global_num_experts=quant_info.global_num_experts,
    )

    has_bias = quant_info.w13_bias is not None or quant_info.w2_bias is not None

    if has_bias:
        assert (
            quant_info.w13_bias is not None and quant_info.w2_bias is not None
        ), "Bias execution requires both w13_bias and w2_bias"
        output = triton_kernel_fused_experts_with_bias(
            hidden_states=hidden_states,
            w1=quant_info.w13_weight,
            w1_pcg=quant_info.w13_precision_config,
            b1=quant_info.w13_bias,
            w2=quant_info.w2_weight,
            w2_pcg=quant_info.w2_precision_config,
            b2=quant_info.w2_bias,
            gemm1_alpha=obj.runner.config.gemm1_alpha,
            gemm1_clamp_limit=obj.runner.config.gemm1_clamp_limit,
            **common_kwargs,
        )
    else:
        output = triton_kernel_fused_experts(
            hidden_states=hidden_states,
            w1=quant_info.w13_weight,
            w2=quant_info.w2_weight,
            **common_kwargs,
        )

    if obj.runner.config.no_combine:
        tokens = dispatch_output.hidden_states.shape[0]
        hidden = dispatch_output.hidden_states.shape[-1]
        total_rows = output.shape[0]
        top_k = total_rows // tokens
        output = output.view(tokens, top_k, hidden)

    return TritonKernelsRunnerOutput(hidden_states=output)

