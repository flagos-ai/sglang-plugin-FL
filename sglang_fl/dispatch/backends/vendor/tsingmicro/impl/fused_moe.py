# TXDA FusedMoE operator implementation.
#
# Delegates to flag_gems.fused.fused_experts_impl which provides a complete
# fused MoE forward pass (moe_align -> GEMM1 -> SiLU+Mul -> GEMM2 -> moe_sum).

from __future__ import annotations

import torch


def fused_moe_txda(
    obj,
    layer: torch.nn.Module,
    dispatch_output,
):
    """TXDA vendor implementation of fused_moe via flag_gems.

    Unpacks the SGLang dispatch_output, delegates to
    ``flag_gems.fused.fused_experts_impl``, and returns a
    ``StandardCombineInput``.
    """
    from flag_gems.fused import fused_experts_impl
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    hidden_states = dispatch_output.hidden_states  # [M, N]
    topk_output = dispatch_output.topk_output       # StandardTopKOutput
    topk_weights, topk_ids, _ = topk_output         # [M, topk], [M, topk], ...

    w1 = layer.w13_weight  # [E, N, 2*I]
    w2 = layer.w2_weight   # [E, N_out, I]

    output = fused_experts_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    return StandardCombineInput(hidden_states=output)
