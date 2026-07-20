# GCU TopK operator implementation.

from __future__ import annotations

from typing import Optional

import torch
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKConfig, _use_aiter, _biased_grouped_topk_postprocess
from sglang.srt.eplb import expert_location_dispatch
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo

def _post_process_topk_ids_gcu(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_config: TopKConfig,
    router_logits: torch.Tensor,
    layer_id: int,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
) -> torch.Tensor:
    topk_ids = _biased_grouped_topk_postprocess(
                topk_ids, expert_location_dispatch_info, num_token_non_padded
            )

    return topk_ids, topk_weights
    
def fused_topk_gcu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: Optional[torch.Tensor] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    scoring_func: str = "softmax",
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)

    if scoring_func == "softmax":
        token_expert_indices = torch.empty(
            M, topk, dtype=torch.int32, device=hidden_states.device
        )
        torch.ops.sgl_kernel.topk_softmax(
            topk_weights,
            topk_ids,
            token_expert_indices,
            gating_output.float(),
        )

        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    elif scoring_func == "sigmoid":
        torch.ops.sgl_kernel.topk_sigmoid(
            topk_weights,
            topk_ids,
            gating_output,
            renormalize,
            correction_bias,
        )
    else:
        raise ValueError(f"Invalid scoring function: {scoring_func}")

    return topk_weights, topk_ids

def select_experts_gcu(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk_config: TopKConfig,
    *,
    layer_id: Optional[int] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
) -> StandardTopKOutput:
    top_k = topk_config.top_k
    renormalize = topk_config.renormalize
    num_fused_shared_experts = topk_config.num_fused_shared_experts
    correction_bias = topk_config.correction_bias

    scoring_func = topk_config.scoring_func

    (
        router_logits,
        correction_bias,
    ) = expert_location_dispatch.transform_select_experts_inputs(
        router_logits=router_logits,
        correction_bias=correction_bias,
        info=expert_location_dispatch_info,
    )

    # DeepSeek V2/V3/R1 series models use grouped_top_k
    # remove num_fused_shared_experts from grouped_topk/biased_grouped_topk
    num_routed_topk = top_k - num_fused_shared_experts

    topk_weights, topk_ids = fused_topk_gcu(
        hidden_states=hidden_states,
        gating_output=router_logits,
        topk=num_routed_topk if _use_aiter else top_k,
        renormalize=renormalize,
        correction_bias=correction_bias,
        scoring_func=scoring_func,
    )

    topk_ids, topk_weights = _post_process_topk_ids_gcu(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        topk_config=topk_config,
        router_logits=router_logits,
        num_token_non_padded=num_token_non_padded,
        layer_id=layer_id,
        expert_location_dispatch_info=expert_location_dispatch_info,
    )

    get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)

    return StandardTopKOutput(topk_weights, topk_ids, router_logits)

def topk_gcu(
    obj,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info=None,
):
    return select_experts_gcu(
        hidden_states=hidden_states,
        layer_id=obj.layer_id,
        router_logits=router_logits,
        topk_config=obj.topk_config,
        num_token_non_padded=num_token_non_padded,
        expert_location_dispatch_info=expert_location_dispatch_info,
    )
