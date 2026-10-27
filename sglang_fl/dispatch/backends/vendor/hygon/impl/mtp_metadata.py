# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Eager speculative metadata for the paged Hygon FlashAttention ABI."""

import torch


def kv_indices_to_page_table(kv_indptr, kv_indices, *, page_size, max_pages, out=None):
    """Convert logical token lists without assuming adjacent physical pages.

    This eager-only conversion deliberately validates every location. Branching
    trees cannot be represented by the vendor's page table and are rejected.
    """
    if page_size <= 0 or max_pages <= 0:
        raise ValueError("page_size and max_pages must be positive")
    if kv_indptr.ndim != 1 or kv_indices.ndim != 1:
        raise ValueError("KV indptr and indices must be one-dimensional")
    if kv_indptr.dtype not in (torch.int32, torch.int64) or kv_indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("KV indptr and indices must be integer tensors")
    if kv_indptr.device != kv_indices.device:
        raise ValueError("KV indptr and indices must be on the same device")
    ptr = kv_indptr.tolist()
    if len(ptr) < 2 or ptr[0] != 0 or any(b < a for a, b in zip(ptr, ptr[1:])):
        raise ValueError("invalid KV indptr")
    if ptr[-1] > kv_indices.numel():
        raise ValueError("KV indptr exceeds the indices capacity")
    lengths = [b - a for a, b in zip(ptr, ptr[1:])]
    if any(n <= 0 or n > page_size * max_pages for n in lengths):
        raise ValueError("KV lengths are empty or exceed page-table capacity")
    shape = (len(lengths), max_pages)
    if out is None:
        out = torch.empty(shape, dtype=torch.int32, device=kv_indices.device)
    if (
        out.shape != shape
        or out.dtype != torch.int32
        or out.device != kv_indices.device
    ):
        raise ValueError("invalid output page-table shape, dtype or device")
    out.fill_(-1)
    for row, (start, end) in enumerate(zip(ptr, ptr[1:])):
        locations = kv_indices[start:end]
        offsets = torch.arange(end - start, device=locations.device) % page_size
        pages = locations // page_size
        expected_pages = pages[::page_size].repeat_interleave(page_size)[: end - start]
        if bool(torch.any(locations < 0)) or not torch.equal(
            locations, expected_pages * page_size + offsets
        ):
            raise ValueError("KV token locations do not form complete aligned pages")
        out[row, : pages[::page_size].numel()] = pages[::page_size].to(torch.int32)
    return out, torch.tensor(lengths, dtype=torch.int32, device=kv_indices.device)


def validate_linear_verify_mask(custom_mask, prefix_lens, width, *, mask_mode=None):
    """Prove a supplied top-k=1 mask equals prefix plus a causal chain."""
    if custom_mask is None:
        return
    if custom_mask.dtype != torch.bool or custom_mask.ndim != 1:
        raise RuntimeError("HCU MTP requires an unpacked boolean chain mask")
    batch_size = len(prefix_lens)
    full_size = sum((prefix + width) * width for prefix in prefix_lens)
    chain_size = batch_size * width * width
    mode_name = getattr(mask_mode, "name", None)
    if mode_name is None:
        mode_name = "FULL_MASK" if custom_mask.numel() >= full_size else "QLEN_ONLY"
    if mode_name not in ("FULL_MASK", "QLEN_ONLY"):
        raise RuntimeError("HCU MTP does not support packed or arbitrary tree masks")
    required = full_size if mode_name == "FULL_MASK" else chain_size
    if custom_mask.numel() < required:
        raise RuntimeError("HCU MTP verify mask is smaller than the query window")
    cursor = 0
    for prefix in prefix_lens:
        columns = prefix + width if mode_name == "FULL_MASK" else width
        actual = custom_mask[cursor : cursor + width * columns].view(width, columns)
        query = torch.arange(width, device=actual.device)[:, None]
        key = torch.arange(columns, device=actual.device)[None, :]
        expected = key <= query + (prefix if mode_name == "FULL_MASK" else 0)
        if not torch.equal(actual, expected):
            raise RuntimeError(
                "HCU MTP requires a linear causal chain mask; tree mask rejected"
            )
        cursor += width * columns
