# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TXDA EAGLE/MTP tree patch: SGLang 0.5.18 contract, rebinding and the torch
greedy verify (CPU only, no device work)."""

import random

import pytest
import torch

eagle_utils = pytest.importorskip("sglang.srt.speculative.eagle_utils")

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches import (  # noqa: E402
    speculative_tree as st,
)


def test_upstream_contract_matches() -> None:
    assert st._signature_is(eagle_utils.sgl_build_tree_kernel_triton, st._TREE_PARAMETERS)
    assert st._signature_is(eagle_utils.verify_tree_greedy_triton, st._VERIFY_PARAMETERS)
    assert st._signature_is(
        eagle_utils.verify_tree_greedy_func, (*st._VERIFY_PARAMETERS, "topk")
    )


def test_apply_binds_triton_build_and_torch_verify(monkeypatch) -> None:
    monkeypatch.setattr(eagle_utils, "sgl_build_tree_kernel_efficient", object(), raising=False)
    monkeypatch.setattr(eagle_utils, "verify_tree_greedy_func", eagle_utils.verify_tree_greedy_func)

    def triton_verify_must_not_run(**kwargs):
        raise AssertionError("SGLang's Triton verify is wrong on txda and must not be used")

    monkeypatch.setattr(eagle_utils, "verify_tree_greedy_triton", triton_verify_must_not_run)
    st._apply()

    assert eagle_utils.sgl_build_tree_kernel_efficient is eagle_utils.sgl_build_tree_kernel_triton
    assert eagle_utils.verify_tree_greedy_func.__wrapped__.__module__ == eagle_utils.__name__

    # chain: every draft correct -> all accepted
    B, N = 1, 4
    predicts = torch.full((B * N,), -7, dtype=torch.int32)
    accept_index = torch.full((B, N), -1, dtype=torch.int32)
    accept_num = torch.zeros(B, dtype=torch.int32)
    out = eagle_utils.verify_tree_greedy_func(
        predicts,
        accept_index,
        accept_num,
        torch.tensor([[11, 21, 31, 41]]),
        torch.arange(N).view(1, N),
        torch.tensor([[1, 2, 3, -1]]),
        torch.full((1, N), -1),
        torch.tensor([[21, 31, 41, 51]]),
        topk=1,
    )
    assert out[0] is predicts and out[1] is accept_index and out[2] is accept_num
    assert accept_index.tolist() == [[0, 1, 2, 3]]
    assert accept_num.tolist() == [3]
    assert predicts.tolist() == [21, 31, 41, 51]


def test_signature_mismatch_leaves_module_untouched(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(eagle_utils, "sgl_build_tree_kernel_efficient", sentinel, raising=False)
    monkeypatch.setattr(eagle_utils, "sgl_build_tree_kernel_triton", lambda x: x)
    st._apply()
    assert eagle_utils.sgl_build_tree_kernel_efficient is sentinel


def _reference(candidates, retrieve_index, nxt, sib, target_predict, steps):
    """Scalar transcription of SGLang's CUDA verify_tree_greedy."""
    B, N = candidates.shape
    flat = target_predict.reshape(-1)
    predicts = torch.full((B * N,), -7, dtype=torch.int32)
    accept_index = torch.full((B, steps), -1, dtype=torch.int32)
    accept_num = torch.zeros(B, dtype=torch.int32)
    for b in range(B):
        last = int(retrieve_index[b, 0])
        accept_index[b, 0] = last
        n_acc, cur = 0, 0
        for _ in range(1, steps):
            cur = int(nxt[b, cur])
            target = int(flat[last])
            found = False
            while cur != -1:
                if int(candidates[b, cur]) == target:
                    predicts[last] = target
                    n_acc += 1
                    accept_index[b, n_acc] = int(retrieve_index[b, cur])
                    last = int(retrieve_index[b, cur])
                    found = True
                    break
                cur = int(sib[b, cur])
            if not found:
                break
        accept_num[b] = n_acc
        predicts[last] = int(flat[last])
    return predicts, accept_index, accept_num


def _random_tree(n, rng):
    """First-child / next-sibling links of a random tree rooted at node 0."""
    nxt, sib, last_child = [-1] * n, [-1] * n, {}
    for i in range(1, n):
        parent = rng.randrange(0, i)
        if parent in last_child:
            sib[last_child[parent]] = i
        else:
            nxt[parent] = i
        last_child[parent] = i
    return nxt, sib


@pytest.mark.parametrize("shape", ["chain", "tree"])
def test_torch_verify_matches_reference(shape) -> None:
    rng = random.Random(0)
    for _ in range(200):
        B, N = rng.randint(1, 4), rng.randint(2, 8)
        steps = rng.randint(2, N)
        links = (
            [([i + 1 for i in range(N - 1)] + [-1], [-1] * N) for _ in range(B)]
            if shape == "chain"
            else [_random_tree(N, rng) for _ in range(B)]
        )
        nxt = torch.tensor([a for a, _ in links])
        sib = torch.tensor([b for _, b in links])
        retrieve_index = torch.arange(B * N).view(B, N)
        candidates = torch.randint(0, 4, (B, N))
        target = torch.randint(0, 4, (B, N))
        want = _reference(candidates, retrieve_index, nxt, sib, target, steps)

        predicts = torch.full((B * N,), -7, dtype=torch.int32)
        accept_index = torch.full((B, steps), -1, dtype=torch.int32)
        accept_num = torch.zeros(B, dtype=torch.int32)
        st._verify_tree_greedy_torch(
            predicts, accept_index, accept_num, candidates, retrieve_index, nxt, sib, target
        )
        assert torch.equal(predicts, want[0])
        assert torch.equal(accept_index, want[1])
        assert torch.equal(accept_num, want[2])
