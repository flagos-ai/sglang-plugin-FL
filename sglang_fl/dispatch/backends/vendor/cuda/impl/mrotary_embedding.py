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

# CUDA vendor MRotaryEmbedding — delegates to SGLang's native implementation.

from __future__ import annotations

from typing import Tuple

import torch


def mrotary_embedding_cuda(
    obj,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MRotaryEmbedding using SGLang's native CUDA/triton kernels.

    For 2D positions: calls forward_triton (triton_mrope_fused).
    For 1D positions: calls parent RotaryEmbedding.forward_cuda logic.
    """
    if positions.ndim == 2 and hasattr(obj, "mrope_section") and obj.mrope_section:
        return obj.forward_triton(positions, query, key)
    # 1D positions: use standard sgl_kernel rope
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

    return RotaryEmbedding.forward_cuda(obj, positions, query, key)
