# Iluvatar rotary embedding operator implementations using SGLang apply_rotary_emb.

from __future__ import annotations

import torch


def rotary_embedding_iluvatar(
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
    Apply rotary position embedding using SGLang apply_rotary_emb.

    Args:
        obj: The calling RotaryEmbedding instance (for interface consistency)
        query: Query tensor [batch, num_heads, head_dim]
        key: Key tensor
        cos: Cosine cache [max_seq_len, rotary_dim // 2]
        sin: Sine cache [max_seq_len, rotary_dim // 2]
        position_ids: Position indices
        rotary_interleaved: Whether to use interleaved rotary (GPT-J style)
        inplace: Whether to modify tensors in-place

    Returns:
        Tuple of (embedded_query, embedded_key)
    """
    from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb

    is_neox_style = not rotary_interleaved
    pos = position_ids.flatten()
    cos_sel = cos.index_select(0, pos)
    sin_sel = sin.index_select(0, pos)

    q = query if inplace else query.clone()
    k = key if inplace else key.clone()

    q_embed = apply_rotary_emb(q, cos_sel, sin_sel, is_neox_style)
    k_embed = apply_rotary_emb(k, cos_sel, sin_sel, is_neox_style)
    return q_embed, k_embed
