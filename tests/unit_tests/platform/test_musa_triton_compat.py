# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Dead PDL branches compile on MUSA; live PDL calls must be rejected."""

from types import SimpleNamespace

import pytest
import torch

triton = pytest.importorskip("triton")
import triton.language as tl

from sglang_fl.dispatch.backends.vendor.mthreads.triton_compat import (
    patch_triton_pdl_symbols,
)


@triton.jit
def _pdl_probe(out, ENABLE_PDL: tl.constexpr):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    tl.store(out, 7)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def test_existing_pdl_symbols_are_preserved(monkeypatch):
    wait, launch = object(), object()
    namespace = SimpleNamespace(gdc_wait=wait, gdc_launch_dependents=launch)
    monkeypatch.setattr(tl.extra, "cuda", namespace, raising=False)
    patch_triton_pdl_symbols()
    patch_triton_pdl_symbols()
    assert namespace.gdc_wait is wait
    assert namespace.gdc_launch_dependents is launch


@pytest.mark.gpu
def test_musa_pdl_dead_branch_and_live_rejection(monkeypatch):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA hardware is required")
    monkeypatch.setattr(tl.extra, "cuda", SimpleNamespace(), raising=False)
    patch_triton_pdl_symbols()
    out = torch.empty(1, device="musa", dtype=torch.int32)
    _pdl_probe[(1,)](out, False)
    assert out.item() == 7
    with pytest.raises(triton.CompilationError) as error:
        _pdl_probe[(1,)](out, True)
    # Triton 3.2 wraps errors from a nested JIT function at the call site.
    cause = error.value
    messages = []
    while cause is not None:
        messages.append(str(cause))
        cause = cause.__cause__
    assert "CUDA PDL is not supported on MUSA" in "\n".join(messages)
