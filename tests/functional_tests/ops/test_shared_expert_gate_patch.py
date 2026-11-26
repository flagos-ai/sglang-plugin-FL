# Copyright (c) 2026 BAAI. All rights reserved.

"""Batch invariance of the shared_expert_gate (N=1 linear) replacement kernel."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sglang_fl.dispatch.shared_expert_gate_patch import (
    batch_invariant_linear_n1,
    make_gate_batch_invariant,
)

pytestmark = [pytest.mark.functional, pytest.mark.gpu]

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("K", [2048, 1000])
def test_rows_do_not_depend_on_batch_size(dtype, K):
    torch.manual_seed(0)
    w = torch.randn(1, K, device="cuda", dtype=dtype) * 0.02
    rows = torch.randn(9, K, device="cuda", dtype=dtype)
    ref = batch_invariant_linear_n1(rows, w)
    filler = torch.randn(1100, K, device="cuda", dtype=dtype)
    # Includes multiples of 8, where cuBLAS switches from gemv to GEMM.
    for M in [9, 15, 16, 64, 123, 128, 528, 1024, 1109]:
        x = torch.cat([filler[: M - rows.shape[0]], rows])
        y = batch_invariant_linear_n1(x, w)[-rows.shape[0]:]
        assert torch.equal(y, ref), f"M={M}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_matches_linear(dtype):
    torch.manual_seed(0)
    x = torch.randn(123, 2048, device="cuda", dtype=dtype)
    w = torch.randn(1, 2048, device="cuda", dtype=dtype) * 0.02
    y = batch_invariant_linear_n1(x, w)
    assert y.shape == (123, 1) and y.dtype == dtype
    ref = (x.double() @ w.double().t()).to(dtype)
    tol = 1e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(y, ref, rtol=tol, atol=tol)


def test_empty_and_3d_input():
    w = torch.randn(1, 64, device="cuda", dtype=torch.bfloat16)
    assert batch_invariant_linear_n1(torch.empty(0, 64, device="cuda", dtype=torch.bfloat16), w).shape == (0, 1)
    x = torch.randn(2, 5, 64, device="cuda", dtype=torch.bfloat16)
    y = batch_invariant_linear_n1(x, w)
    assert y.shape == (2, 5, 1)
    assert torch.equal(y.reshape(10, 1), batch_invariant_linear_n1(x.reshape(10, 64), w))


def test_make_gate_batch_invariant_only_patches_n1_no_bias():
    gate = torch.nn.Linear(64, 1, bias=False).cuda().to(torch.bfloat16)
    assert make_gate_batch_invariant(gate)
    x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(gate(x), batch_invariant_linear_n1(x, gate.weight))
    assert not make_gate_batch_invariant(torch.nn.Linear(64, 2, bias=False))
    assert not make_gate_batch_invariant(torch.nn.Linear(64, 1, bias=True))
    assert not make_gate_batch_invariant(None)
    # CPU input falls back to F.linear
    cpu_gate = torch.nn.Linear(8, 1, bias=False)
    make_gate_batch_invariant(cpu_gate)
    xc = torch.randn(3, 8)
    assert torch.equal(cpu_gate(xc), F.linear(xc, cpu_gate.weight))
