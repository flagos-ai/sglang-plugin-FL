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

# Ascend MRotaryEmbedding operator implementation.
# Ports MRotaryEmbedding.forward_npu (sglang/srt/layers/rotary_embedding/mrope.py).

from __future__ import annotations

from typing import Tuple

import torch


def mrotary_embedding_ascend(
    obj,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    import torch_npu

    # npu_mrope has an internal size limit; fall back to pure-torch for large embeddings.
    if query.shape[1] > 4096:
        return obj.forward_native(positions, query, key, None)

    rotary_mode = "half" if obj.is_neox_style else "interleave"
    # [0,0,0] tells the kernel to derive multimodal sections from positions.ndim.
    mrope_section = [0, 0, 0]

    return torch_npu.npu_mrope(
        positions,
        query,
        key,
        obj.cos_sin_cache,
        obj.head_size,
        mrope_section=mrope_section,
        rotary_mode=rotary_mode,
    )
