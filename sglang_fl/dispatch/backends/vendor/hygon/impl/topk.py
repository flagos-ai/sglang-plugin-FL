# Bridge: TopK
#
# SGLang signature: forward_cuda(self, hidden_states, router_logits, *,
#                                num_token_non_padded=None,
#                                expert_location_dispatch_info=None) -> TopKOutput
# Dispatch signature: fn(obj, hidden_states, router_logits, **kwargs) -> TopKOutput
# Mapping: pass-through (1:1)

from __future__ import annotations

from typing import Optional

import torch


def topk_hcu(
    obj,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info=None,
):
    """
    TopK routing using SGLang's native select_experts (STANDARD output format).

    Args:
        obj: The TopK instance
        hidden_states: Input tensor
        router_logits: Router logits for expert selection
        num_token_non_padded: Optional number of non-padded tokens
        expert_location_dispatch_info: Optional expert location dispatch info

    Returns:
        TopKOutput (StandardTopKOutput format)
    """
    from sglang.srt.layers.moe.topk import select_experts

    # Use STANDARD output format (not TRITON_KERNEL or BYPASSED)
    # This matches SGLang's forward_cuda STANDARD path
    obj.topk_config.torch_native = False
    return select_experts(
        hidden_states=hidden_states,
        layer_id=obj.layer_id,
        router_logits=router_logits,
        topk_config=obj.topk_config,
        num_token_non_padded=num_token_non_padded,
        expert_location_dispatch_info=expert_location_dispatch_info,
    )
