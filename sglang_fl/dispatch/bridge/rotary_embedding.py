# Bridge: RotaryEmbedding
#
# SGLang signature:
#   forward_cuda(self, positions, query, key, offsets=None,
#                fused_set_kv_buffer_arg=None)
#     -> tuple[Tensor, Tensor]
#
# Dispatch signatures:
#   rotary_embedding:
#     fn(obj, query, key, cos, sin, position_ids, rotary_interleaved=False,
#        inplace=True)
#     -> tuple[Tensor, Tensor]
#
#   rotary_embedding_with_kv_cache:
#     fn(obj, query, key, cos, sin, position_ids, fused_set_kv_buffer_arg,
#        rotary_interleaved=False)
#     -> tuple[Tensor, Tensor]
#
# SGLang-specific handling:
#   - Extract cos/sin from self.cos_sin_cache (shape [max_seq_len, rotary_dim])
#   - Handle offsets (add to positions)
#   - Handle partial rotary_dim (only apply to first rotary_dim dimensions)
#   - fused_set_kv_buffer_arg: dispatch as separate fused op
#     "rotary_embedding_with_kv_cache" (fused RoPE + KV cache write)
#   - Reshape query/key from [batch, num_heads*head_size] to [batch, num_heads, head_size]

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang_fl.dispatch import call_op


def rotary_embedding_bridge(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
    fused_set_kv_buffer_arg=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SGLang RotaryEmbedding forward → dispatch call_op("rotary_embedding", ...).

    Handles SGLang-specific parameter translation before delegating to dispatch.
    When fused_set_kv_buffer_arg is provided, dispatches to the fused
    "rotary_embedding_with_kv_cache" op instead.
    """
    # Handle offsets
    if offsets is not None:
        positions = positions + offsets

    # Extract cos_sin_cache from self, match dtype to query (SGLang stores as fp32)
    cos_sin_cache = self.cos_sin_cache.to(dtype=query.dtype)
    cos, sin = cos_sin_cache.chunk(2, dim=-1)

    # Determine rotary parameters from self
    head_size = getattr(self, "head_size", query.shape[-1])
    is_neox_style = getattr(self, "is_neox_style", True)
    rotary_interleaved = not is_neox_style

    # SGLang query/key come in as [batch, num_heads * head_size]
    # Reshape to [batch, num_heads, head_size] for dispatch
    batch_size = positions.size(0)
    query_shape = query.shape
    key_shape = key.shape

    query = query.view(batch_size, -1, head_size)
    key = key.view(batch_size, -1, head_size)

    # Choose op based on whether fused KV cache write is requested
    if fused_set_kv_buffer_arg is not None:
        op_name = "rotary_embedding_with_kv_cache"
        op_kwargs = dict(
            rotary_interleaved=rotary_interleaved,
        )
    else:
        op_name = "rotary_embedding"
        op_kwargs = dict(
            rotary_interleaved=rotary_interleaved,
            inplace=True,
        )

    # If rotary_dim < head_size, only apply to first rotary_dim dimensions
    rotary_dim = cos.shape[-1] * 2  # cos is half_dim, full rotary_dim = 2 * half_dim
    if rotary_dim < head_size:
        query_rot = query[..., :rotary_dim]
        key_rot = key[..., :rotary_dim]
        query_pass = query[..., rotary_dim:]
        key_pass = key[..., rotary_dim:]

        if fused_set_kv_buffer_arg is not None:
            q_embed, k_embed = call_op(
                op_name,
                self,
                query_rot,
                key_rot,
                cos,
                sin,
                positions,
                fused_set_kv_buffer_arg,
                **op_kwargs,
            )
        else:
            q_embed, k_embed = call_op(
                op_name,
                self,
                query_rot,
                key_rot,
                cos,
                sin,
                positions,
                **op_kwargs,
            )

        query = torch.cat((q_embed, query_pass), dim=-1)
        key = torch.cat((k_embed, key_pass), dim=-1)
    else:
        if fused_set_kv_buffer_arg is not None:
            q_embed, k_embed = call_op(
                op_name,
                self,
                query,
                key,
                cos,
                sin,
                positions,
                fused_set_kv_buffer_arg,
                **op_kwargs,
            )
        else:
            q_embed, k_embed = call_op(
                op_name,
                self,
                query,
                key,
                cos,
                sin,
                positions,
                **op_kwargs,
            )
        query = q_embed
        key = k_embed

    # Reshape back to original shapes
    query = query.reshape(query_shape)
    key = key.reshape(key_shape)

    return query, key
