# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TXDA KV-release hardening on SGLang 0.5.18 Req/ChunkCache shapes (CPU only)."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("sglang.srt.mem_cache.chunk_cache")

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches import (  # noqa: E402
    kv_leak_fix as kv,
)


class _Allocator:
    def __init__(self):
        self.freed = []

    def free(self, pages):
        self.freed.extend(int(p) for p in pages)


def _cache(row):
    return SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=torch.tensor([row])),
        token_to_kv_pool_allocator=_Allocator(),
    )


def _req(**kw):
    base = dict(
        req_pool_idx=0,
        rid="r0",
        cache_protected_len=0,
        kv_committed_len=0,
        kv=SimpleNamespace(kv_allocated_len=0),
        origin_input_ids=[1, 2, 3],
        output_ids=[],
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_release_frees_upstream_range_and_filters_zero_holes() -> None:
    cache = _cache([5, 6, 0, 7, 9, 9])
    req = _req(cache_protected_len=1, kv_committed_len=4, kv=SimpleNamespace(kv_allocated_len=4))
    kv._safe_cache_finished_req(cache, req, kv_len_to_handle=4)
    # [cache_protected_len, kv_len_to_handle) without the 0 hole.
    assert cache.token_to_kv_pool_allocator.freed == [6, 7]


def test_repaired_safety_net_reads_req_kv_allocated_len() -> None:
    # commit=0 / alloc=0 but the row still owns pages: the REPAIRED branch,
    # which crashed on 0.5.18 when it read req.kv_allocated_len.
    cache = _cache([3, 4, 0, 0])
    req = _req()
    assert not hasattr(req, "kv_allocated_len")
    kv._safe_cache_finished_req(cache, req, kv_len_to_handle=0)
    assert cache.token_to_kv_pool_allocator.freed == [3, 4]
    assert req._kv_leak_repaired is True


def test_unsafe_stale_tail_is_left_alone() -> None:
    cache = _cache([3, 4, 0, 8])  # page beyond prompt+output bound (3) -> reused slot
    req = _req()
    kv._safe_cache_finished_req(cache, req, kv_len_to_handle=0)
    assert cache.token_to_kv_pool_allocator.freed == []


def test_kv_len_to_handle_is_required() -> None:
    with pytest.raises(TypeError):
        kv._safe_cache_finished_req(_cache([1]), _req())
