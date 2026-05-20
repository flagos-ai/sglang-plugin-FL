# MUSA rotary embedding operator implementations.
#
# TODO: Replace NotImplementedError with torch_musa native kernel once verified
# on hardware. Current behavior: falls back to flaggems (Triton) or reference.

from __future__ import annotations

import torch


def rotary_embedding_musa(
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
    Apply rotary position embedding on MUSA.

    Args:
        obj: The calling nn.Module (unused, for interface consistency)
        query: Query tensor [num_tokens, num_heads, head_dim]
        key: Key tensor [num_tokens, num_kv_heads, head_dim]
        cos: Cosine cache [max_seq_len, rotary_dim // 2]
        sin: Sine cache [max_seq_len, rotary_dim // 2]
        position_ids: Position indices [num_tokens]
        rotary_interleaved: Whether to use interleaved rotary
        inplace: Whether to modify tensors in-place

    Returns:
        Tuple of (embedded_query, embedded_key)
    """
    raise NotImplementedError(
        "rotary_embedding: no torch_musa kernel wired yet; falling back to flaggems/reference"
    )
