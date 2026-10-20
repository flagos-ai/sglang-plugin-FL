# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        query,
        key,
        cache,
        positions,
        axis_map,
        query_stride,
        key_stride,
        position_stride,
        query_heads: tl.constexpr,
        key_heads: tl.constexpr,
        head_size: tl.constexpr,
        rotary_dim: tl.constexpr,
        padded_query_heads: tl.constexpr,
        padded_key_heads: tl.constexpr,
        padded_head_size: tl.constexpr,
        SECTION_T: tl.constexpr,
        SECTION_H: tl.constexpr,
        SECTION_W: tl.constexpr,
        INTERLEAVED: tl.constexpr,
        INTERLEAVED_GLM: tl.constexpr,
        NEOX: tl.constexpr,
    ):
        token = tl.program_id(0)
        query += token * query_stride
        key += token * key_stride
        half = rotary_dim // 2
        position_t = tl.load(positions + token)
        position_h = tl.load(positions + position_stride + token)
        position_w = tl.load(positions + 2 * position_stride + token)
        offsets = tl.arange(0, padded_head_size // 2)
        if INTERLEAVED:
            if INTERLEAVED_GLM:
                axes = tl.load(axis_map + offsets, mask=offsets < half)
                mask_t, mask_h, mask_w = axes == 0, axes == 1, axes == 2
            else:
                mask_h = (offsets % 3 == 1) & (offsets <= 3 * SECTION_H)
                mask_w = (offsets % 3 == 2) & (offsets <= 3 * SECTION_W)
                mask_t = ~(mask_h | mask_w)
        else:
            end_t = SECTION_T
            end_h = end_t + SECTION_H
            mask_t = offsets < end_t
            mask_h = (end_t <= offsets) & (offsets < end_h)
            mask_w = (end_h <= offsets) & (offsets < half)

        cache_t = cache + position_t * rotary_dim
        cache_h = cache + position_h * rotary_dim
        cache_w = cache + position_w * rotary_dim
        cos_t = tl.load(cache_t + offsets, mask=mask_t, other=0.0)
        sin_t = tl.load(cache_t + half + offsets, mask=mask_t, other=0.0)
        cos_h = tl.load(cache_h + offsets, mask=mask_h, other=0.0)
        sin_h = tl.load(cache_h + half + offsets, mask=mask_h, other=0.0)
        cos_w = tl.load(cache_w + offsets, mask=mask_w, other=0.0)
        sin_w = tl.load(cache_w + half + offsets, mask=mask_w, other=0.0)
        cos = cos_t + cos_h + cos_w
        sin = sin_t + sin_h + sin_w

        if NEOX:
            qh = tl.arange(0, padded_query_heads)[:, None]
            kh = tl.arange(0, padded_key_heads)[:, None]
            dim = tl.arange(0, padded_head_size // 2)[None, :]
            query_mask = (qh < query_heads) & (dim < half)
            key_mask = (kh < key_heads) & (dim < half)
            query_first = qh * head_size + dim
            key_first = kh * head_size + dim
            query_second = query_first + half
            key_second = key_first + half
        else:
            qh = tl.arange(0, padded_query_heads)[:, None]
            kh = tl.arange(0, padded_key_heads)[:, None]
            pair = tl.arange(0, padded_head_size // 2)[None, :]
            query_mask = (qh < query_heads) & (pair < half)
            key_mask = (kh < key_heads) & (pair < half)
            query_first = qh * head_size + 2 * pair
            key_first = kh * head_size + 2 * pair
            query_second = query_first + 1
            key_second = key_first + 1

        q1 = tl.load(query + query_first, mask=query_mask, other=0.0)
        q2 = tl.load(query + query_second, mask=query_mask, other=0.0)
        k1 = tl.load(key + key_first, mask=key_mask, other=0.0)
        k2 = tl.load(key + key_second, mask=key_mask, other=0.0)
        tl.store(query + query_first, q1 * cos - q2 * sin, mask=query_mask)
        tl.store(query + query_second, q2 * cos + q1 * sin, mask=query_mask)
        tl.store(key + key_first, k1 * cos - k2 * sin, mask=key_mask)
        tl.store(key + key_second, k2 * cos + k1 * sin, mask=key_mask)

    return kernel


def mrotary_embedding(obj, positions, query, key):
    import triton

    tokens = query.shape[0]
    query_heads = query.numel() // (tokens * obj.head_size)
    key_heads = key.numel() // (tokens * obj.head_size)
    section = obj.mrope_section
    axis_map = obj.axis_map if obj.axis_map is not None else obj.cos_sin_cache
    _kernel()[(tokens,)](
        query,
        key,
        obj.cos_sin_cache,
        positions,
        axis_map,
        query.stride(0),
        key.stride(0),
        positions.stride(0),
        query_heads=query_heads,
        key_heads=key_heads,
        head_size=obj.head_size,
        rotary_dim=obj.rotary_dim,
        padded_query_heads=triton.next_power_of_2(query_heads),
        padded_key_heads=triton.next_power_of_2(key_heads),
        padded_head_size=triton.next_power_of_2(obj.head_size),
        SECTION_T=section[0],
        SECTION_H=section[1],
        SECTION_W=section[2],
        INTERLEAVED=obj.mrope_interleaved,
        INTERLEAVED_GLM=obj.mrope_interleaved_glm,
        NEOX=obj.is_neox_style,
        num_warps=4,
        num_stages=1,
    )
    return query, key
