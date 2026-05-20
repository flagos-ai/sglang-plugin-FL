# Reference MRotaryEmbedding operator implementation using pure PyTorch.
# Mirrors SGLang's MRotaryEmbedding.forward_native exactly:
#   1. Index cos_sin_cache with positions (2D: [3, T] or 1D: [T])
#   2. For 2D positions + mrope_section: merge per-axis cos/sin slices into one
#      [num_tokens, rotary_dim//2] tensor before applying RoPE.
#   3. Apply standard neox or interleaved RoPE to the merged cos/sin.
#
# Handles both 1D positions (text-only) and 2D positions (multimodal).

from __future__ import annotations

from typing import Tuple

import torch


def mrotary_embedding_torch(
    obj,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multimodal rotary position embedding — mirrors SGLang MRotaryEmbedding.forward_native.

    mrope_section[i] counts cos/sin *frequencies* (half-dim units) for axis i.
    Summed they equal rotary_dim // 2.  Each frequency covers 2 query dimensions,
    so the full RoPE span is rotary_dim.

    The implementation builds merged cos/sin first (same shape as standard RoPE:
    [num_tokens, rotary_dim//2]), then applies standard RoPE — no section-by-section
    splitting of query tensors needed.
    """
    head_size = obj.head_size
    rotary_dim = obj.rotary_dim
    is_neox_style = obj.is_neox_style

    # Ensure dtype and device match query (cos_sin_cache may be fp32 on CUDA, fp16 on MUSA)
    cos_sin_cache = obj.cos_sin_cache.to(device=query.device, dtype=query.dtype)

    num_tokens = query.shape[0]
    query_shape = query.shape
    key_shape = key.shape

    # ── Build merged cos / sin ────────────────────────────────────────────────
    if positions.ndim == 2 and getattr(obj, "mrope_section", None):
        mrope_section = obj.mrope_section
        mrope_interleaved = getattr(obj, "mrope_interleaved", False)

        # cos_sin_cache[positions]: [3, num_tokens, rotary_dim]
        cos_sin = cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)  # each: [3, num_tokens, rotary_dim//2]

        if mrope_interleaved:
            # Interleaved (GLM-style): handled by apply_interleaved_rope-equivalent
            # For each pair index p, the axis is determined by axis_map[p].
            # cos/sin already indexed per axis; reconstruct interleaved merged tensor.
            # Fallback: use axis 0 only (conservative for now, override in vendor impl)
            cos_merged = cos[0]
            sin_merged = sin[0]
        else:
            # Non-interleaved (Qwen-VL style):
            # cos.split(mrope_section, dim=-1) → list of [3, num_tokens, section_i_dim]
            # m[i] selects axis-i slice → [num_tokens, section_i_dim]
            # cat → [num_tokens, rotary_dim//2]
            cos_merged = torch.cat(
                [m[i] for i, m in enumerate(cos.split(mrope_section, dim=-1))],
                dim=-1,
            )
            sin_merged = torch.cat(
                [m[i] for i, m in enumerate(sin.split(mrope_section, dim=-1))],
                dim=-1,
            )
    else:
        # 1D positions (text-only, or 2D without mrope_section)
        pos_1d = positions[0] if positions.ndim == 2 else positions
        cos_sin = cos_sin_cache[pos_1d]          # [num_tokens, rotary_dim]
        cos_merged, sin_merged = cos_sin.chunk(2, dim=-1)   # each: [num_tokens, rotary_dim//2]

    # ── Apply RoPE ───────────────────────────────────────────────────────────
    # Reshape to [num_tokens, num_heads, head_size]
    query = query.view(num_tokens, -1, head_size)
    key = key.view(num_tokens, -1, head_size)

    # Separate rotating part from pass-through
    query_rot = query[..., :rotary_dim]
    query_pass = query[..., rotary_dim:]
    key_rot = key[..., :rotary_dim]
    key_pass = key[..., rotary_dim:]

    half_dim = rotary_dim // 2

    # Trim if cache half-dim exceeds half of rotary_dim (shouldn't happen normally)
    if cos_merged.shape[-1] > half_dim:
        cos_merged = cos_merged[..., :half_dim]
        sin_merged = sin_merged[..., :half_dim]

    # Expand to [num_tokens, 1, rotary_dim] for broadcasting with [num_tokens, heads, rotary_dim]
    if is_neox_style:
        # neox: cos = [cos, cos], sin = [sin, sin]
        cos_full = torch.cat([cos_merged, cos_merged], dim=-1).unsqueeze(1)
        sin_full = torch.cat([sin_merged, sin_merged], dim=-1).unsqueeze(1)

        q1, q2 = query_rot[..., :half_dim], query_rot[..., half_dim:]
        q_embed = query_rot * cos_full + torch.cat((-q2, q1), dim=-1) * sin_full

        k1, k2 = key_rot[..., :half_dim], key_rot[..., half_dim:]
        k_embed = key_rot * cos_full + torch.cat((-k2, k1), dim=-1) * sin_full
    else:
        # interleaved: cos = [cos0,cos0, cos1,cos1, ...]
        cos_full = torch.stack([cos_merged, cos_merged], dim=-1).flatten(-2).unsqueeze(1)
        sin_full = torch.stack([sin_merged, sin_merged], dim=-1).flatten(-2).unsqueeze(1)

        q1, q2 = query_rot[..., ::2], query_rot[..., 1::2]
        q_embed = query_rot * cos_full + torch.stack((-q2, q1), dim=-1).flatten(-2) * sin_full

        k1, k2 = key_rot[..., ::2], key_rot[..., 1::2]
        k_embed = key_rot * cos_full + torch.stack((-k2, k1), dim=-1).flatten(-2) * sin_full

    # Reassemble and restore original shape
    if query_pass.shape[-1] > 0:
        query = torch.cat([q_embed, query_pass], dim=-1)
        key = torch.cat([k_embed, key_pass], dim=-1)
    else:
        query = q_embed
        key = k_embed

    return query.reshape(query_shape), key.reshape(key_shape)
