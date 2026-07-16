# FlagOS TopK operator implementation.

from __future__ import annotations

from typing import Optional

import flag_gems
import torch
from sglang.srt.layers.moe.topk import StandardTopKOutput


def topk_flagos(
    obj,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info=None,
):
    """
    TopK routing using flag_gems.topk_softmax.

    Reference: vllm-plugin-FL/vllm_fl/ops/_fl_ops.py

    Args:
        obj: The TopK instance
        hidden_states: Input tensor
        router_logits: Router logits for expert selection
        num_token_non_padded: Optional number of non-padded tokens
        expert_location_dispatch_info: Optional expert location dispatch info

    Returns:
        TopKOutput (StandardTopKOutput format)
    """
    topk = obj.topk_config.top_k
    renormalize = obj.topk_config.renormalize

    M = hidden_states.shape[0]
    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    token_expert_indices = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    flag_gems.topk_softmax(
        topk_weights, topk_ids, token_expert_indices, router_logits, renormalize
    )

    return StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=router_logits,
    )
