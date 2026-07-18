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
    topk = obj.topk_config.topk
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
        token_expert_indices=token_expert_indices,
    )
