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

"""SGLang 0.5.18 Ascend TopK adapter."""

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
    """Delegate to the native NPU method to retain 0.5.18 routing semantics."""

    forward_npu = getattr(obj, "forward_npu", None)
    if forward_npu is None:
        raise RuntimeError(
            "SGLang 0.5.18 TopK.forward_npu is required by the Ascend adapter"
        )
    return forward_npu(
        hidden_states,
        router_logits,
        num_token_non_padded=num_token_non_padded,
        expert_location_dispatch_info=expert_location_dispatch_info,
    )
