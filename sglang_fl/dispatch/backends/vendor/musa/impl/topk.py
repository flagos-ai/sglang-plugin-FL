# MUSA TopK operator implementation using sglang's built-in MUSA Triton kernels.
#
# sglang upstream ships topk_softmax / topk_sigmoid Triton kernels specifically
# for MUSA in: sglang.srt.hardware_backend.musa.kernels.topk
#
# Handles:
#   - scoring_func="softmax"  → topk_softmax_triton_kernel
#   - scoring_func="sigmoid"  → topk_sigmoid_triton_kernel
#   - grouped topk / custom routing → delegated to select_experts (torch native)

from __future__ import annotations

from typing import Optional

import torch


def topk_musa(
    obj,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info=None,
):
    """
    TopK expert routing on MUSA via Triton kernels.

    Uses sglang's upstream MUSA-specific topk_softmax / topk_sigmoid Triton
    kernels for the common non-grouped case.  Falls back to select_experts
    (torch-native path) for grouped topk, custom routing functions, or fused
    shared experts.

    Args:
        obj: The TopK instance (provides obj.topk_config, obj.layer_id)
        hidden_states: Input tensor [num_tokens, hidden_size]
        router_logits: Gating logits [num_tokens, num_experts]
        num_token_non_padded: Optional non-padded token count
        expert_location_dispatch_info: Optional EP dispatch info

    Returns:
        StandardTopKOutput(topk_weights, topk_ids, router_logits)
    """
    from sglang.srt.hardware_backend.musa.kernels.topk import (
        topk_sigmoid,
        topk_softmax,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput, select_experts

    topk_cfg = obj.topk_config

    # Delegate complex routing to the torch-native path in select_experts.
    if (
        topk_cfg.use_grouped_topk
        or topk_cfg.custom_routing_function is not None
        or topk_cfg.num_fused_shared_experts > 0
    ):
        topk_cfg.torch_native = True
        return select_experts(
            hidden_states=hidden_states,
            layer_id=obj.layer_id,
            router_logits=router_logits,
            topk_config=topk_cfg,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )

    topk = topk_cfg.top_k
    renormalize = topk_cfg.renormalize
    scoring_func = topk_cfg.scoring_func
    correction_bias = topk_cfg.correction_bias

    M = hidden_states.shape[0]
    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=hidden_states.device)
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)

    # Triton kernels expect float32 gating logits.
    gating = router_logits.float()

    if scoring_func == "softmax":
        moe_softcapping = float(getattr(topk_cfg, "moe_softcapping", 0.0) or 0.0)
        topk_softmax(
            topk_weights,
            topk_ids,
            gating,
            renormalize,
            moe_softcapping,
            correction_bias,
        )
    elif scoring_func == "sigmoid":
        topk_sigmoid(
            topk_weights,
            topk_ids,
            gating,
            renormalize,
            correction_bias,
        )
    else:
        # Unknown scoring function: fall back to torch-native select_experts.
        topk_cfg.torch_native = True
        return select_experts(
            hidden_states=hidden_states,
            layer_id=obj.layer_id,
            router_logits=router_logits,
            topk_config=topk_cfg,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )

    return StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=router_logits,
    )
