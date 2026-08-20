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

# Ascend rotary embedding operator implementations.

from __future__ import annotations

import torch


def rotary_embedding_ascend(
    obj,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool = False,
    inplace: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding using Ascend NPU.

    Args:
        obj: The calling obj (for interface consistency)
        query: Query tensor [num_tokens, num_heads, rotary_dim]
        key: Key tensor [num_tokens, num_kv_heads, rotary_dim]
        cos: Cosine cache [max_seq_len, rotary_dim // 2]
        sin: Sine cache [max_seq_len, rotary_dim // 2]
        position_ids: Position indices [num_tokens]
        rotary_interleaved: Whether to use interleaved rotary
        inplace: Whether to modify tensors in-place

    Returns:
        Tuple of (embedded_query, embedded_key)
    """
    import torch_npu

    num_tokens = query.shape[0]
    rotary_dim = query.shape[-1]

    # Reconstruct cos_sin_cache from separate cos and sin
    cos_sin_cache = torch.cat([cos, sin], dim=-1)

    # Save original shapes
    query_shape = query.shape
    key_shape = key.shape

    # Flatten for NPU kernel: [num_tokens, num_heads * rotary_dim]
    query_flat = query.contiguous().view(num_tokens, -1)
    key_flat = key.contiguous().view(num_tokens, -1)

    is_neox_style = not rotary_interleaved

    torch_npu._npu_rotary_embedding(
        position_ids,
        query_flat,
        key_flat,
        rotary_dim,
        cos_sin_cache,
        is_neox_style,
    )

    q_embed = query_flat.view(query_shape)
    k_embed = key_flat.view(key_shape)

    return q_embed, k_embed
