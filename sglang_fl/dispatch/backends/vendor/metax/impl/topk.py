# MetaX TopK routing implementations.

from __future__ import annotations

from typing import Optional

import torch


def topk_metax(
    obj,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info=None,
):
    from sglang_fl.dispatch.backends.vendor.cuda.impl.topk import topk_cuda

    return topk_cuda(
        obj,
        hidden_states,
        router_logits,
        num_token_non_padded=num_token_non_padded,
        expert_location_dispatch_info=expert_location_dispatch_info,
    )
