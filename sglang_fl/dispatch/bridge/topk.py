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

from sglang_fl.dispatch import call_op


def topk_bridge(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info=None,
):
    """SGLang TopK forward → dispatch call_op("topk", ...)."""
    return call_op(
        "topk",
        self,
        hidden_states,
        router_logits,
        num_token_non_padded=num_token_non_padded,
        expert_location_dispatch_info=expert_location_dispatch_info,
    )
