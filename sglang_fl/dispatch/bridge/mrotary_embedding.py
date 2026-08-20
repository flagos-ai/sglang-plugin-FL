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

# Bridge: MRotaryEmbedding
#
# SGLang signature:
#   forward_cuda(self, positions, query, key, fused_set_kv_buffer_arg=None)
#     -> tuple[Tensor, Tensor]
#
# Dispatch signature:
#   fn(obj, positions, query, key) -> tuple[Tensor, Tensor]
#
# SGLang-specific handling:
#   - positions can be 1D [num_tokens] or 2D [3, num_tokens] (multimodal)
#   - mrope_section splits rotary_dim across 3 axes (text/image/video)
#   - fused_set_kv_buffer_arg: not supported by dispatch, fall through to native

from __future__ import annotations

from typing import Tuple

import torch

from sglang_fl.dispatch import call_op


def mrotary_embedding_bridge(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    fused_set_kv_buffer_arg=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SGLang MRotaryEmbedding forward → dispatch call_op("mrotary_embedding", ...)."""
    # fused_set_kv_buffer_arg requires native kernel — fall through
    if fused_set_kv_buffer_arg is not None:
        return self.forward_native(positions, query, key, fused_set_kv_buffer_arg)

    return call_op("mrotary_embedding", self, positions, query, key)
