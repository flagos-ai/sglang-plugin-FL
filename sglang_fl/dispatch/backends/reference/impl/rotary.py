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

# Reference rotary embedding operator implementations using PyTorch.

from __future__ import annotations

import torch


def rotary_embedding_torch(
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
    Apply rotary position embedding using PyTorch.

    Args:
        obj: The calling obj (for interface consistency)
        query: Query tensor
        key: Key tensor
        cos: Cosine cache [max_seq_len, rotary_dim]
        sin: Sine cache [max_seq_len, rotary_dim]
        position_ids: Position indices
        rotary_interleaved: Whether to use interleaved rotary
        inplace: Whether to modify tensors in-place (ignored in reference impl)

    Returns:
        Tuple of (embedded_query, embedded_key)
    """
    # Get cos/sin for the positions
    if position_ids.dim() == 1:
        cos_selected = cos[position_ids]
        sin_selected = sin[position_ids]
    else:
        cos_selected = cos[position_ids]
        sin_selected = sin[position_ids]

    # Expand dimensions to match query/key shape
    if query.dim() == 4:
        # [batch, num_heads, seq_len, head_dim]
        cos_selected = cos_selected.unsqueeze(1)
        sin_selected = sin_selected.unsqueeze(1)
    elif query.dim() == 3:
        # [seq_len, num_heads, head_dim]
        cos_selected = cos_selected.unsqueeze(1)
        sin_selected = sin_selected.unsqueeze(1)

    # Check if we need to repeat cos/sin to match head_dim
    rotary_dim = cos_selected.shape[-1]
    head_dim = query.shape[-1]

    if rotary_dim != head_dim:
        cos_selected = torch.cat([cos_selected, cos_selected], dim=-1)
        sin_selected = torch.cat([sin_selected, sin_selected], dim=-1)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    if rotary_interleaved:

        def rotate_interleaved(x):
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            return torch.stack((-x2, x1), dim=-1).flatten(-2)

        q_embed = (query * cos_selected) + (rotate_interleaved(query) * sin_selected)
        k_embed = (key * cos_selected) + (rotate_interleaved(key) * sin_selected)
    else:
        q_embed = (query * cos_selected) + (rotate_half(query) * sin_selected)
        k_embed = (key * cos_selected) + (rotate_half(key) * sin_selected)

    return q_embed, k_embed
