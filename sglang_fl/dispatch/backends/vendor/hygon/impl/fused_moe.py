# Bridge: UnquantizedFusedMoEMethod
#
# SGLang signature: forward_cuda(self, layer, dispatch_output) -> CombineInput
# Dispatch signature: fn(obj, layer, dispatch_output) -> CombineInput
# Mapping: pass-through (1:1)

from __future__ import annotations

import torch


def fused_moe_hcu(
    obj,
    layer: torch.nn.Module,
    dispatch_output,
):
    """
    Fused MoE expert computation using SGLang's native Triton kernels.

    This implementation delegates to SGLang's MoeRunner with the TRITON backend,
    which uses the standard (non-triton_kernels, non-flashinfer) path.

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
