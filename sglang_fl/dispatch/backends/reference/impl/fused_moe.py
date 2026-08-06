# Reference implementation for fused_moe using SGLang's native Triton MoE runner.

from __future__ import annotations

import torch


def fused_moe_reference(
    obj,
    layer: torch.nn.Module,
    dispatch_output,
):
    """
    Reference fused MoE implementation using SGLang's native Triton kernels.

    This delegates to SGLang's TritonMoeRunner — which is Triton-based and works
    on any platform with a Triton backend (NVIDIA, Ascend, MUSA, etc.).

    Args:
        obj: The UnquantizedFusedMoEMethod instance
        layer: The MoE layer module
        dispatch_output: StandardDispatchOutput containing hidden_states and topk_output

    Returns:
        CombineInput (StandardCombineInput)
    """
    from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo

    quant_info = TritonMoeQuantInfo(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        b13=getattr(layer, "w13_weight_bias", None),
        b2=getattr(layer, "w2_weight_bias", None),
    )
    return obj.runner.run(dispatch_output, quant_info)
