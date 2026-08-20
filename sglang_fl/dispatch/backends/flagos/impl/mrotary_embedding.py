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

# FlagOS MRotaryEmbedding — triton-based implementation that works on all dtypes.
#
# Uses SGLang's built-in triton kernels (forward_triton / forward_native) which
# handle bfloat16/float16/float32 correctly, unlike the JIT C++ rope kernel that
# has dtype restrictions for certain rope_dim values on OOT platforms.
#
# This issue is fixed upstream in SGLang v0.5.13 (commit 6a3316dd1e) by skipping
# the JIT kernel when is_out_of_tree()=True. For v0.5.11, this flagos implementation
# provides the same workaround via the dispatch layer.

from __future__ import annotations

from typing import Tuple

import torch


def mrotary_embedding_flagos(
    obj,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MRotaryEmbedding using SGLang's triton-based kernels.

    For 2D positions (multimodal): calls forward_triton (triton_mrope_fused).
    For 1D positions (text/MTP decode): calls forward_native (triton fallback).

    Both paths handle all dtypes (bfloat16, float16, float32) correctly.
    """
    if positions.ndim == 2 and hasattr(obj, "mrope_section") and obj.mrope_section:
        return obj.forward_triton(positions, query, key)
    return obj.forward_native(positions, query, key)
