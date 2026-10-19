# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    # Only the plain softmax top-k kernel is replaced; sglang's own pre/post
    # processing (EPLB input transform, padding mask, recorder) is kept. Other
    # routing modes raise so the dispatcher falls back to the next backend.
    from sglang.srt.eplb import expert_location_dispatch
    from sglang.srt.eplb.expert_distribution import (
        get_global_expert_distribution_recorder,
    )
    from sglang.srt.layers.moe.topk import _post_process_topk_ids

    cfg = obj.topk_config
    if (
        cfg.use_grouped_topk
        or cfg.custom_routing_function is not None
        or cfg.correction_bias is not None
        or cfg.scoring_func != "softmax"
        or cfg.num_fused_shared_experts
        or cfg.apply_routed_scaling_factor_on_output
    ):
        raise NotImplementedError("flagos topk: only plain softmax top-k supported")

    router_logits, _ = expert_location_dispatch.transform_select_experts_inputs(
        router_logits=router_logits,
        correction_bias=None,
        info=expert_location_dispatch_info,
    )

    M = hidden_states.shape[0]
    topk = cfg.top_k
    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    token_expert_indices = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    flag_gems.topk_softmax(
        topk_weights, topk_ids, token_expert_indices, router_logits, cfg.renormalize
    )

    topk_ids, topk_weights = _post_process_topk_ids(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        topk_config=cfg,
        router_logits=router_logits,
        num_token_non_padded=num_token_non_padded,
        layer_id=obj.layer_id,
        expert_location_dispatch_info=expert_location_dispatch_info,
    )
    get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)

    return StandardTopKOutput(topk_weights, topk_ids, router_logits)
