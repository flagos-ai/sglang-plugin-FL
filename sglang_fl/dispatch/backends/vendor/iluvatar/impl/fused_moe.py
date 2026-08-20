# Iluvatar FusedMoE operator implementation using SGLang's native fused_experts.

from __future__ import annotations

import torch


def _moe_filter_expert(obj) -> bool:
    cfg = obj.moe_runner_config
    return (
        cfg.num_experts is None
        or cfg.num_local_experts is None
        or cfg.num_experts != cfg.num_local_experts
    )


def fused_moe_iluvatar(
    obj,
    layer: torch.nn.Module,
    dispatch_output,
):
    if _moe_filter_expert(obj):
        raise NotImplementedError(
            "fused_moe_iluvatar; falling back to flaggems/reference"
        )

    from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo

    quant_info = TritonMoeQuantInfo(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        b13=getattr(layer, "w13_weight_bias", None),
        b2=getattr(layer, "w2_weight_bias", None),
    )
    return obj.runner.run(dispatch_output, quant_info)
