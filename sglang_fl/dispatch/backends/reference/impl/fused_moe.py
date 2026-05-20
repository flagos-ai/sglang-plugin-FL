# Reference FusedMoE implementation via SGLang's native Triton MoE runner.

from __future__ import annotations

import torch


def fused_moe_torch(
    obj,
    layer: torch.nn.Module,
    dispatch_output,
):
    from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo

    quant_info = TritonMoeQuantInfo(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        b13=getattr(layer, "w13_weight_bias", None),
        b2=getattr(layer, "w2_weight_bias", None),
    )
    return obj.runner.run(dispatch_output, quant_info)
