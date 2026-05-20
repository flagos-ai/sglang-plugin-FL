# CUDA rotary embedding operator implementations using sgl_kernel.

from __future__ import annotations

import torch


def rotary_embedding_cuda(
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
    Apply rotary position embedding using sgl_kernel.

    Args:
        obj: The calling obj (for interface consistency)
        query: Query tensor
        key: Key tensor
        cos: Cosine cache
        sin: Sine cache
        position_ids: Position indices
        rotary_interleaved: Whether to use interleaved rotary
        inplace: Whether to modify tensors in-place

    Returns:
        Tuple of (embedded_query, embedded_key)
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


def rotary_embedding_with_kv_cache_cuda(
    obj,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    fused_set_kv_buffer_arg,
    rotary_interleaved: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused RoPE + KV cache write using sgl_kernel.

    Single kernel applies rotary embedding to q/k and writes rotated k + v to KV cache.

    Args:
        obj: The calling obj (for interface consistency)
        query: Query tensor [num_tokens, num_heads, head_dim]
        key: Key tensor [num_tokens, num_kv_heads, head_dim]
        cos: Cosine cache
        sin: Sine cache
        position_ids: Position indices
        fused_set_kv_buffer_arg: FusedSetKVBufferArg or dict with KV cache info
        rotary_interleaved: Whether to use interleaved rotary

    Returns:
        Tuple of (embedded_query, embedded_key)
    """
    from sglang.jit_kernel.rope import apply_rope_with_cos_sin_cache_inplace

    # Reconstruct cos_sin_cache from separate cos and sin
    cos_sin_cache = torch.cat([cos, sin], dim=-1)
    is_neox = not rotary_interleaved

    apply_rope_with_cos_sin_cache_inplace(
        positions=position_ids,
        q=query,
        k=key,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
        fused_args=fused_set_kv_buffer_arg,
    )
    return query, key
