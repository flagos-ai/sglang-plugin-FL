# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TXDA FusedMoE: FlagGems ``_tsingmicro`` fused experts with full MoE semantics.

``flag_gems.fused_experts_impl`` is rebound by FlagGems to the ``_tsingmicro``
vendor kernel (moe_align -> GEMM1 -> SiLU*Mul -> GEMM2 -> moe_sum); the generic
``flag_gems.fused.fused_experts_impl`` is a different function, so the
top-level name is used on purpose.

The generic FlagOS ``fused_moe`` backend is only enabled on CoreX, so on TXDA
this vendor op is the one dispatch selects. It reuses the FlagOS config check
(unsupported setups raise ``NotImplementedError`` and dispatch falls back
instead of computing a wrong result) and forwards the same arguments:
activation, router weight on input, global expert count, biases and the
routed scaling factor.
"""

from __future__ import annotations

import torch

from sglang_fl.dispatch.backends.flagos.impl.fused_moe import _validate_supported_config


def fused_moe_txda(obj, layer: torch.nn.Module, dispatch_output):
    """TXDA vendor implementation of the unquantized fused_moe bridge op."""
    _validate_supported_config(obj, layer, dispatch_output)
    from flag_gems import fused_experts_impl  # _tsingmicro vendor kernel
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    config = obj.moe_runner_config
    topk = dispatch_output.topk_output
    output = fused_experts_impl(
        hidden_states=dispatch_output.hidden_states,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_weights=topk.topk_weights,
        topk_ids=topk.topk_ids,
        activation=config.activation,
        apply_router_weight_on_input=config.apply_router_weight_on_input,
        global_num_experts=config.num_experts,
        w1_bias=getattr(layer, "w13_weight_bias", None),
        w2_bias=getattr(layer, "w2_weight_bias", None),
    )
    factor = config.routed_scaling_factor
    if factor is not None and factor != 1.0:
        output.mul_(factor)
    return StandardCombineInput(hidden_states=output)
