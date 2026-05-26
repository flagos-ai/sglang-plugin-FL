# Ascend TopK operator implementation.
# Delegates to sglang's fused_topk_npu (hardware_backend/npu/moe/topk.py).

from __future__ import annotations

from typing import Optional

import torch


def topk_ascend(
    obj,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info=None,
):
    
    from sglang.srt.hardware_backend.npu.moe.topk import fused_topk_npu

    return fused_topk_npu(
        hidden_states=hidden_states,
        router_logits=router_logits,
        topk_config=obj.topk_config,
        num_token_non_padded=num_token_non_padded,
        expert_location_dispatch_info=expert_location_dispatch_info,
        layer_id=obj.layer_id,
    )
