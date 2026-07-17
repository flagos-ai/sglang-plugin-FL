# Bridge: RotaryEmbedding
#
# SGLang signature:
#   forward_cuda(self, positions, query, key, offsets=None,
#                fused_set_kv_buffer_arg=None)
#     -> tuple[Tensor, Tensor]
#
# Dispatch signature:
#   fn(obj, query, key, cos, sin, position_ids, rotary_interleaved=False,
#      inplace=True)
#     -> tuple[Tensor, Tensor]
#
# SGLang-specific handling:
#   - Extract cos/sin from self.cos_sin_cache (shape [max_seq_len, rotary_dim])
#   - Handle offsets (add to positions)
#   - Handle partial rotary_dim (only apply to first rotary_dim dimensions)
#   - fused_set_kv_buffer_arg: not supported by dispatch, fall through to native
#   - Reshape query/key from [batch, num_heads*head_size] to [batch, num_heads, head_size]

from __future__ import annotations

from typing import Optional, Tuple

import torch



def rotary_embedding_hcu(
    obj,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool = False,
    inplace: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SGLang RotaryEmbedding forward → dispatch call_op("rotary_embedding", ...).

    Handles SGLang-specific parameter translation before delegating to dispatch.
    """
    from sgl_kernel import rotary_embedding as sgl_rotary_embedding

    sgl_rotary_embedding(
        position_ids,
        query,
        key,
        cos,
        sin,
        rotary_interleaved,
    )
    return query, key
