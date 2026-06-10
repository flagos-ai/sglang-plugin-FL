# FlagOS FusedMoE operator implementation.

from __future__ import annotations

import torch


def _standard_topk_to_triton_kernels(topk_weights, topk_ids, n_expts_tot):
    """Convert StandardTopKOutput tensors → (RoutingData, GatherIndx, ScatterIndx).

    This mirrors the second half of ``triton_kernels.routing.routing_torch``:
    given already-computed topk_weights (float32, [T, k]) and topk_ids
    (int32, [T, k]), build the sorted routing structures that
    ``matmul_ogs`` expects.
    """
    from triton_kernels.routing import (
        GatherIndx,
        RoutingData,
        ScatterIndx,
        compute_expt_data_torch,
    )

    n_tokens, n_expts_act = topk_weights.shape
    n_gates_pad = n_tokens * n_expts_act

    # -- sort each token's selections by expert (ascending expert id) ------
    expt_indx_2d, sort_indices = torch.sort(topk_ids, dim=1)
    expt_scal_2d = torch.gather(topk_weights, 1, sort_indices)

    # -- flatten to 1-D [T*k] ---------------------------------------------
    expt_scal = expt_scal_2d.reshape(-1)
    expt_indx = expt_indx_2d.reshape(-1).to(torch.int32)

    # -- sort by expert so tokens for the same expert are contiguous -------
    topk_indx = torch.argsort(expt_indx, stable=True).to(torch.int32)
    gate_indx = torch.argsort(topk_indx, stable=True).to(torch.int32)
    gate_scal = expt_scal[topk_indx]

    hist = torch.histc(
        expt_indx.float(), bins=n_expts_tot, max=n_expts_tot - 1
    ).int()

    expt_data = compute_expt_data_torch(hist, n_expts_tot, n_gates_pad)

    routing_data = RoutingData(gate_scal, hist, n_expts_tot, n_expts_act, expt_data)
    gather_indx = GatherIndx(src_indx=topk_indx, dst_indx=gate_indx)
    scatter_indx = ScatterIndx(src_indx=gate_indx, dst_indx=topk_indx)
    return routing_data, gather_indx, scatter_indx


def fused_moe_flagos(
    obj,
    layer: torch.nn.Module,
    dispatch_output,
):
    from sglang.srt.layers.moe.moe_runner.triton_kernels import (
        TritonKernelsQuantInfo,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    from flaggems_sglang.fused_moe_kernel import (
        triton_kernel_fused_experts,
        triton_kernel_fused_experts_with_bias,
    )

    # SGLang stores weights as [E, N, K] but matmul_ogs expects [E, K, N].
    # Transpose the last two dimensions to match the expected layout.
    w13 = layer.w13_weight.transpose(-1, -2).contiguous()
    w2 = layer.w2_weight.transpose(-1, -2).contiguous()

    quant_info = TritonKernelsQuantInfo(
        w13_weight=w13,
        w2_weight=w2,
        w13_bias=getattr(layer, "w13_weight_bias", None),
        w2_bias=getattr(layer, "w2_weight_bias", None),
    )

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output

    # --- adapt topk format ------------------------------------------------
    if TopKOutputChecker.format_is_triton_kernels(topk_output):
        # Already in triton_kernels format (RoutingData, GatherIndx, ScatterIndx)
        routing_data, gather_indx, scatter_indx = topk_output
    else:
        # StandardTopKOutput → convert on the fly
        routing_data, gather_indx, scatter_indx = _standard_topk_to_triton_kernels(
            topk_output.topk_weights,
            topk_output.topk_ids,
            n_expts_tot=obj.runner.config.num_experts,
        )

    common_kwargs = dict(
        routing_data=routing_data,
        gather_indx=gather_indx,
        scatter_indx=None if obj.runner.config.no_combine else scatter_indx,
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

    if (
        obj.runner.config.routed_scaling_factor is not None
        and obj.runner.config.routed_scaling_factor != 1.0
        and not obj.runner.config.no_combine
    ):
        output.mul_(obj.runner.config.routed_scaling_factor)

    return StandardCombineInput(hidden_states=output)

