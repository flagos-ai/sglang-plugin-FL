# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TXDA speculative KV write locations match SGLang's kernel contract (CPU only)."""

import pytest
import torch

cache_locs = pytest.importorskip("sglang.kernels.ops.speculative.cache_locs")

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches import (  # noqa: E402
    speculative_cache_locs as scl,
)


def _req_to_token():
    # 4 pool rows x 16 slots, distinct values
    return (torch.arange(4 * 16, dtype=torch.int32) + 100).view(4, 16)


def _reference(req_pool_indices, req_to_token, start, end, bs, n):
    out = torch.full((bs * n,), -1, dtype=torch.int64)
    pt = 0
    for i in range(bs):
        seg = req_to_token[int(req_pool_indices[i]), int(start[i]) : int(end[i])]
        out[pt : pt + seg.numel()] = seg.to(torch.int64)
        pt += seg.numel()
    return out, pt


def test_uniform_matches_row_slices() -> None:
    r2t = _req_to_token()
    rpi = torch.tensor([2, 0, 3])
    start = torch.tensor([5, 0, 9])
    out = scl.assign_extend_cache_locs_uniform_func(rpi, r2t, start, 3, 4, "cpu")
    ref, _ = _reference(rpi, r2t, start, start + 4, 3, 4)
    assert out.dtype == torch.int64
    assert torch.equal(out, ref)


def test_variable_lengths_are_packed_back_to_back() -> None:
    r2t = _req_to_token()
    rpi = torch.tensor([1, 3, 0])
    start = torch.tensor([2, 7, 4])
    end = torch.tensor([5, 8, 4])  # lengths 3, 1, 0
    out = scl.assign_extend_cache_locs_func(rpi, r2t, start, end, 3, 4, "cpu")
    ref, used = _reference(rpi, r2t, start, end, 3, 4)
    assert out.shape == (12,)
    assert torch.equal(out[:used], ref[:used])


def test_apply_replaces_module_functions(monkeypatch) -> None:
    monkeypatch.setattr(cache_locs, "assign_extend_cache_locs_func", lambda *a, **k: None)
    monkeypatch.setattr(cache_locs, "assign_extend_cache_locs_uniform_func", lambda *a, **k: None)
    monkeypatch.setattr(scl.assign_extend_cache_locs_func, "_txda", False, raising=False)
    scl._apply()
    assert cache_locs.assign_extend_cache_locs_func is scl.assign_extend_cache_locs_func
    assert cache_locs.assign_extend_cache_locs_uniform_func is scl.assign_extend_cache_locs_uniform_func


def test_assign_req_to_token_pool_writes_every_slot() -> None:
    r2t = _req_to_token()
    before = r2t.clone()
    rpi = torch.tensor([2, 0, 3])
    start = torch.tensor([5, 0, 9])
    end = start + torch.tensor([3, 0, 4])
    ocl = torch.arange(500, 507, dtype=torch.int64)
    scl.assign_req_to_token_pool_func(rpi, r2t, start, end, ocl, 3)
    want = before.clone()
    want[2, 5:8] = torch.tensor([500, 501, 502], dtype=torch.int32)
    want[3, 9:13] = torch.tensor([503, 504, 505, 506], dtype=torch.int32)
    assert torch.equal(r2t, want)


def test_assign_extend_cache_locs_launch_matches_kernel_contract() -> None:
    r2t = _req_to_token()
    rpi = torch.tensor([1, 3])
    start = torch.tensor([2, 7])
    end = torch.tensor([4, 10])  # 2 + 3 accepted tokens
    out = torch.zeros(8, dtype=torch.int64)
    scl._AssignExtendCacheLocsLaunch()[(2,)](rpi, r2t, start, end, out, 16, 2)
    ref, used = _reference(rpi, r2t, start, end, 2, 4)
    assert torch.equal(out[:used], ref[:used])
