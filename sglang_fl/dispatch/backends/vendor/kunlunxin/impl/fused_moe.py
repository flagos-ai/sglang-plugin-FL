# Kunlunxin (Baidu Kunlun XPU) FusedMoE operator implementation.
#
# Unquantized (bf16/fp16) expert MLP delegated to the fused kernel
#   torch.ops.sgl_kernel.klx_fused_experts
# which performs the expert half of the MoE on the XPU:
#   gather per-token experts -> gate_up GEMM -> SiLU-and-mul -> down GEMM
#   -> weighted top-k combine (using the supplied topk_weights).
#
# Routing (softmax/topk, grouped-topk, correction bias, renormalize, routed
# scaling) is NOT done here: sglang's own select_experts already produces the
# materialized topk_weights / topk_ids (on the Kunlunxin cuda-compat device that
# path itself uses sgl_kernel.topk_softmax), so we just consume them. This is why
# we use klx_fused_experts (topk-in) rather than klx_fused_moe (router-logits-in):
# no need to recover the routing config, and grouped/biased routing is handled
# correctly upstream.
#
# Weight layout (sglang UnquantizedFusedMoEMethod.create_weights):
#   w13_weight [E, 2*inter, hidden]  (gate = first half, up = second half)
#   w2_weight  [E, hidden, inter]
# which is exactly the [E, 2N, K] / [E, K, N] convention klx_fused_experts expects.

from __future__ import annotations

import torch


def fused_moe_kunlunxin(
    obj,
    layer: torch.nn.Module,
    dispatch_output,
):
    """Unquantized fused MoE expert computation on Kunlunxin Platform.

    Args:
        obj: The UnquantizedFusedMoEMethod instance (holds moe_runner_config).
        layer: The MoE layer module (holds w13_weight / w2_weight).
        dispatch_output: StandardDispatchOutput (hidden_states, .., topk_output).

    Returns:
        StandardCombineInput(hidden_states=final [num_tokens, hidden]).
    """
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    # Deferred, kunlunxin-local side-effect import (registers torch.ops.sgl_kernel.*).
    import sgl_kernel  # noqa: F401

    x = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    topk_weights = topk_output.topk_weights
    topk_ids = topk_output.topk_ids

    # sglang's select_experts emits topk_ids as int64 (Long), but the klx
    # klx_fused_experts kernel requires int32 (Int) — else it raises
    # "expected scalar type Int but found Long". Cast defensively.
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)

    activation = obj.moe_runner_config.activation
    inplace = obj.moe_runner_config.inplace
    is_gated = obj.moe_runner_config.is_gated

    out = torch.ops.sgl_kernel.klx_fused_experts.default(
        x,  # hidden_states
        layer.w13_weight,  # w1 [E, 2*inter, hidden]
        layer.w2_weight,  # w2 [E, hidden, inter]
        topk_weights,  # precomputed by sglang select_experts
        topk_ids,
        inplace,
        activation,
        is_gated,
    )

    return StandardCombineInput(hidden_states=out)
