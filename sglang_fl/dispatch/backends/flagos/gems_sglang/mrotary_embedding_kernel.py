"""MRotaryEmbedding Triton Kernel — v8.

v7 → v8: introduce head-grouped 2D-grid kernel with GROUP_SIZE to balance SM
utilization and memory efficiency.

Analysis of v5 vs v7 trade-off:
- v5 (split-head): 0.971x wgeo, great decode-1 (0.964x), but 18x redundant
  cos/sin loads for decode-1 (one per head)
- v7 (single-token): 0.967x wgeo, worse decode-1 (0.944x), but 1 cos/sin load
  — SM under-utilization for small N

v8 synthesis: head-grouped kernel inspired by unsloth reference. Each program
handles GROUP_SIZE consecutive heads (default 4). For decode-1 (18 heads):
- v5: 18 programs, 18 cos/sin loads
- v7: 1 program, 1 cos/sin load
- v8: ceil(18/4)=5 programs, 5 cos/sin loads

Grid: (num_tokens, num_groups) where num_groups = ceil((n_qh+n_kh)/GROUP_SIZE)
Each program loads cos/sin once, then rotates GROUP_SIZE heads (first n_qh from
q, remainder from k). Reduces redundant loads by GROUP_SIZE vs v5, improves
SM util by num_groups vs v7.

GROUP_SIZE=4 chosen as sweet spot per unsloth (larger can hurt from reduced
occupancy). Use for small N (<=128) where SM util matters; keep v7's single-
token kernel for larger N where parallelism is natural.

Pure Triton, signatures unchanged. Bit-equivalent semantics to sglang's
`triton_mrope_fused`.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _mrope_fused_fwd_single_token(
    q_ptr,
    k_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_stride0,
    k_stride0,
    pos_stride0,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_rd: tl.constexpr,
    section_t: tl.constexpr,
    section_h: tl.constexpr,
    section_w: tl.constexpr,
    is_interleaved: tl.constexpr,
    is_interleaved_glm: tl.constexpr,
    is_neox_style: tl.constexpr,
    axis_map_ptr,
):
    """Original v7 single-token kernel (one program per token, all heads)."""
    pid = tl.program_id(0)
    q_row = q_ptr + pid * q_stride0
    k_row = k_ptr + pid * k_stride0
    half_rd: tl.constexpr = rd // 2
    pad_rd_half: tl.constexpr = pad_rd // 2

    t_pos = tl.load(positions_ptr + 0 * pos_stride0 + pid)
    h_pos = tl.load(positions_ptr + 1 * pos_stride0 + pid)
    w_pos = tl.load(positions_ptr + 2 * pos_stride0 + pid)

    pair_idx = tl.arange(0, pad_rd_half)
    valid_pair = pair_idx < half_rd

    if is_interleaved:
        if is_interleaved_glm:
            axes = tl.load(axis_map_ptr + pair_idx, mask=valid_pair, other=0)
            pos = tl.where(axes == 1, h_pos, tl.where(axes == 2, w_pos, t_pos))
        else:
            mod3 = pair_idx % 3
            is_h = (mod3 == 1) & (pair_idx <= 3 * section_h)
            is_w = (mod3 == 2) & (pair_idx <= 3 * section_w)
            pos = tl.where(is_h, h_pos, tl.where(is_w, w_pos, t_pos))
    else:
        t_end: tl.constexpr = section_t
        h_end: tl.constexpr = section_t + section_h
        is_h = (pair_idx >= t_end) & (pair_idx < h_end)
        is_w = pair_idx >= h_end
        pos = tl.where(is_h, h_pos, tl.where(is_w, w_pos, t_pos))

    cos_base = cos_sin_cache_ptr + pos * rd
    cos_row = tl.load(cos_base + pair_idx, mask=valid_pair, other=0.0).to(tl.float32)
    sin_row = tl.load(cos_base + pair_idx + half_rd, mask=valid_pair, other=0.0).to(tl.float32)

    q_heads = tl.arange(0, pad_n_qh)
    k_heads = tl.arange(0, pad_n_kh)

    if is_neox_style:
        even_off = pair_idx
        odd_off = pair_idx + half_rd
    else:
        even_off = pair_idx * 2
        odd_off = even_off + 1

    q_even_addr = q_heads[:, None] * hd + even_off[None, :]
    q_odd_addr = q_heads[:, None] * hd + odd_off[None, :]
    q_mask = (q_heads[:, None] < n_qh) & (pair_idx[None, :] < half_rd)

    qe = tl.load(q_row + q_even_addr, mask=q_mask, other=0.0).to(cos_row.dtype)
    qo = tl.load(q_row + q_odd_addr, mask=q_mask, other=0.0).to(cos_row.dtype)
    qe_new = qe * cos_row[None, :] - qo * sin_row[None, :]
    qo_new = qo * cos_row[None, :] + qe * sin_row[None, :]
    tl.store(q_row + q_even_addr, qe_new, mask=q_mask)
    tl.store(q_row + q_odd_addr, qo_new, mask=q_mask)

    k_even_addr = k_heads[:, None] * hd + even_off[None, :]
    k_odd_addr = k_heads[:, None] * hd + odd_off[None, :]
    k_mask = (k_heads[:, None] < n_kh) & (pair_idx[None, :] < half_rd)

    ke = tl.load(k_row + k_even_addr, mask=k_mask, other=0.0).to(cos_row.dtype)
    ko = tl.load(k_row + k_odd_addr, mask=k_mask, other=0.0).to(cos_row.dtype)
    ke_new = ke * cos_row[None, :] - ko * sin_row[None, :]
    ko_new = ko * cos_row[None, :] + ke * sin_row[None, :]
    tl.store(k_row + k_even_addr, ke_new, mask=k_mask)
    tl.store(k_row + k_odd_addr, ko_new, mask=k_mask)


@triton.jit
def _mrope_fused_fwd_head_grouped(
    q_ptr,
    k_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_stride0,
    k_stride0,
    pos_stride0,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rd: tl.constexpr,
    pad_rd: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    section_t: tl.constexpr,
    section_h: tl.constexpr,
    section_w: tl.constexpr,
    is_interleaved: tl.constexpr,
    is_interleaved_glm: tl.constexpr,
    is_neox_style: tl.constexpr,
    axis_map_ptr,
):
    """Head-grouped 2D-grid kernel: (token_id, group_id) -> GROUP_SIZE heads."""
    token_id = tl.program_id(0)
    group_id = tl.program_id(1)

    q_row = q_ptr + token_id * q_stride0
    k_row = k_ptr + token_id * k_stride0
    half_rd: tl.constexpr = rd // 2
    pad_rd_half: tl.constexpr = pad_rd // 2

    # Load positions once per (token, group)
    t_pos = tl.load(positions_ptr + 0 * pos_stride0 + token_id)
    h_pos = tl.load(positions_ptr + 1 * pos_stride0 + token_id)
    w_pos = tl.load(positions_ptr + 2 * pos_stride0 + token_id)

    pair_idx = tl.arange(0, pad_rd_half)
    valid_pair = pair_idx < half_rd

    # Compute per-pair axis selection
    if is_interleaved:
        if is_interleaved_glm:
            axes = tl.load(axis_map_ptr + pair_idx, mask=valid_pair, other=0)
            pos = tl.where(axes == 1, h_pos, tl.where(axes == 2, w_pos, t_pos))
        else:
            mod3 = pair_idx % 3
            is_h = (mod3 == 1) & (pair_idx <= 3 * section_h)
            is_w = (mod3 == 2) & (pair_idx <= 3 * section_w)
            pos = tl.where(is_h, h_pos, tl.where(is_w, w_pos, t_pos))
    else:
        t_end: tl.constexpr = section_t
        h_end: tl.constexpr = section_t + section_h
        is_h = (pair_idx >= t_end) & (pair_idx < h_end)
        is_w = pair_idx >= h_end
        pos = tl.where(is_h, h_pos, tl.where(is_w, w_pos, t_pos))

    # Load cos/sin once per group
    cos_base = cos_sin_cache_ptr + pos * rd
    cos_row = tl.load(cos_base + pair_idx, mask=valid_pair, other=0.0).to(tl.float32)
    sin_row = tl.load(cos_base + pair_idx + half_rd, mask=valid_pair, other=0.0).to(tl.float32)

    # Compute head indices for this group
    base_head = group_id * GROUP_SIZE
    local_heads = tl.arange(0, GROUP_SIZE)
    global_heads = base_head + local_heads

    total_heads: tl.constexpr = n_qh + n_kh
    valid_head = global_heads < total_heads

    # Separate Q and K heads
    is_q_head = global_heads < n_qh
    q_head_idx = global_heads
    k_head_idx = global_heads - n_qh

    if is_neox_style:
        even_off = pair_idx
        odd_off = pair_idx + half_rd
    else:
        even_off = pair_idx * 2
        odd_off = even_off + 1

    # Process Q heads (vectorized across valid Q heads in the group)
    q_even_addr = q_head_idx[:, None] * hd + even_off[None, :]
    q_odd_addr = q_head_idx[:, None] * hd + odd_off[None, :]
    q_mask = (is_q_head[:, None]) & (valid_head[:, None]) & (pair_idx[None, :] < half_rd)

    qe = tl.load(q_row + q_even_addr, mask=q_mask, other=0.0).to(cos_row.dtype)
    qo = tl.load(q_row + q_odd_addr, mask=q_mask, other=0.0).to(cos_row.dtype)
    qe_new = qe * cos_row[None, :] - qo * sin_row[None, :]
    qo_new = qo * cos_row[None, :] + qe * sin_row[None, :]
    tl.store(q_row + q_even_addr, qe_new, mask=q_mask)
    tl.store(q_row + q_odd_addr, qo_new, mask=q_mask)

    # Process K heads (vectorized across valid K heads in the group)
    is_k_head = ~is_q_head
    k_even_addr = k_head_idx[:, None] * hd + even_off[None, :]
    k_odd_addr = k_head_idx[:, None] * hd + odd_off[None, :]
    k_mask = (is_k_head[:, None]) & (valid_head[:, None]) & (pair_idx[None, :] < half_rd)

    ke = tl.load(k_row + k_even_addr, mask=k_mask, other=0.0).to(cos_row.dtype)
    ko = tl.load(k_row + k_odd_addr, mask=k_mask, other=0.0).to(cos_row.dtype)
    ke_new = ke * cos_row[None, :] - ko * sin_row[None, :]
    ko_new = ko * cos_row[None, :] + ke * sin_row[None, :]
    tl.store(k_row + k_even_addr, ke_new, mask=k_mask)
    tl.store(k_row + k_odd_addr, ko_new, mask=k_mask)


@triton.jit
def _rope_1d_fwd(
    q_ptr,
    k_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_stride0,
    k_stride0,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_rd: tl.constexpr,
    is_neox_style: tl.constexpr,
):
    pid = tl.program_id(0)
    q_row = q_ptr + pid * q_stride0
    k_row = k_ptr + pid * k_stride0
    half_rd: tl.constexpr = rd // 2
    pad_rd_half: tl.constexpr = pad_rd // 2

    pos = tl.load(positions_ptr + pid)
    pair_idx = tl.arange(0, pad_rd_half)
    valid_pair = pair_idx < half_rd

    cos = tl.load(cos_sin_cache_ptr + pos * rd + pair_idx, mask=valid_pair, other=0.0).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + pos * rd + pair_idx + half_rd, mask=valid_pair, other=0.0
    ).to(tl.float32)

    q_heads = tl.arange(0, pad_n_qh)
    k_heads = tl.arange(0, pad_n_kh)

    if is_neox_style:
        even_off = pair_idx
        odd_off = pair_idx + half_rd
    else:
        even_off = pair_idx * 2
        odd_off = even_off + 1

    q_even_addr = q_heads[:, None] * hd + even_off[None, :]
    q_odd_addr = q_heads[:, None] * hd + odd_off[None, :]
    q_mask = (q_heads[:, None] < n_qh) & (pair_idx[None, :] < half_rd)
    qe = tl.load(q_row + q_even_addr, mask=q_mask, other=0.0).to(cos.dtype)
    qo = tl.load(q_row + q_odd_addr, mask=q_mask, other=0.0).to(cos.dtype)
    tl.store(q_row + q_even_addr, qe * cos[None, :] - qo * sin[None, :], mask=q_mask)
    tl.store(q_row + q_odd_addr, qo * cos[None, :] + qe * sin[None, :], mask=q_mask)

    k_even_addr = k_heads[:, None] * hd + even_off[None, :]
    k_odd_addr = k_heads[:, None] * hd + odd_off[None, :]
    k_mask = (k_heads[:, None] < n_kh) & (pair_idx[None, :] < half_rd)
    ke = tl.load(k_row + k_even_addr, mask=k_mask, other=0.0).to(cos.dtype)
    ko = tl.load(k_row + k_odd_addr, mask=k_mask, other=0.0).to(cos.dtype)
    tl.store(k_row + k_even_addr, ke * cos[None, :] - ko * sin[None, :], mask=k_mask)
    tl.store(k_row + k_odd_addr, ko * cos[None, :] + ke * sin[None, :], mask=k_mask)




def triton_mrope_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    mrope_section: List[int],
    head_size: int,
    rotary_dim: int,
    mrope_interleaved: bool,
    mrope_interleaved_glm: bool,
    is_neox_style: bool,
    axis_map: Optional[torch.Tensor],
) -> None:
    """In-place mrotary embedding on q and k. Matches sglang's signature."""
    num_tokens = q.shape[0]
    n_qh = q.shape[1] // head_size
    n_kh = k.shape[1] // head_size
    pad_n_qh = triton.next_power_of_2(n_qh)
    pad_n_kh = triton.next_power_of_2(n_kh)
    pad_rd = triton.next_power_of_2(rotary_dim)

    if cos_sin_cache.dtype != q.dtype or cos_sin_cache.device != q.device:
        cos_sin_cache = cos_sin_cache.to(device=q.device, dtype=q.dtype)

    axis_map_arg = axis_map if axis_map is not None else q

    # Dispatch: use head-grouped kernel for small N to improve SM utilization
    if num_tokens <= 128:
        GROUP_SIZE = 4
        total_heads = n_qh + n_kh
        num_groups = (total_heads + GROUP_SIZE - 1) // GROUP_SIZE

        _mrope_fused_fwd_head_grouped[(num_tokens, num_groups)](
            q,
            k,
            cos_sin_cache,
            positions,
            q.stride(0),
            k.stride(0),
            positions.stride(0),
            n_qh,
            n_kh,
            head_size,
            rotary_dim,
            pad_rd,
            GROUP_SIZE,
            mrope_section[0],
            mrope_section[1],
            mrope_section[2],
            mrope_interleaved,
            mrope_interleaved_glm,
            is_neox_style,
            axis_map_arg,
            num_warps=4,
        )
    else:
        # Use single-token kernel for larger N (natural parallelism)
        _mrope_fused_fwd_single_token[(num_tokens,)](
            q,
            k,
            cos_sin_cache,
            positions,
            q.stride(0),
            k.stride(0),
            positions.stride(0),
            n_qh,
            n_kh,
            head_size,
            rotary_dim,
            pad_n_qh,
            pad_n_kh,
            pad_rd,
            mrope_section[0],
            mrope_section[1],
            mrope_section[2],
            mrope_interleaved,
            mrope_interleaved_glm,
            is_neox_style,
            axis_map_arg,
            num_warps=4,
        )


def _rope_1d(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    is_neox_style: bool,
) -> None:
    if positions.ndim == 2:
        positions = positions[0].contiguous()
    num_tokens = q.shape[0]
    n_qh = q.shape[1] // head_size
    n_kh = k.shape[1] // head_size
    pad_n_qh = triton.next_power_of_2(n_qh)
    pad_n_kh = triton.next_power_of_2(n_kh)
    pad_rd = triton.next_power_of_2(rotary_dim)
    if cos_sin_cache.dtype != q.dtype or cos_sin_cache.device != q.device:
        cos_sin_cache = cos_sin_cache.to(device=q.device, dtype=q.dtype)
    _rope_1d_fwd[(num_tokens,)](
        q,
        k,
        cos_sin_cache,
        positions,
        q.stride(0),
        k.stride(0),
        n_qh,
        n_kh,
        head_size,
        rotary_dim,
        pad_n_qh,
        pad_n_kh,
        pad_rd,
        is_neox_style,
    )
