# Copyright (c) 2026 BAAI. All rights reserved.

"""Functional correctness tests for the Xing4 mHC dispatch operators.

Xing4.0-29B-A4B routes its mHC pre/post pair through ``call_op`` on NPU. These
tests exercise that real dispatch entry point and compare the result against the
reference backend, which implements the arithmetic of the checkpoint's own
modeling file independently.

Every sinkhorn iteration divides by a sum, so rounding error from one iteration
is amplified rather than averaged out. The tolerances below are derived from that
structure (see ``_TOL``) instead of being loosened until the comparison passes.
"""

from __future__ import annotations

import pytest
import torch

from sglang_fl.dispatch import call_op

pytestmark = [pytest.mark.functional, pytest.mark.gpu]


HC_MULT = 4
HIDDEN = 3584
RMS_EPS = 1e-6
SINKHORN_EPS = 1e-6
POST_MULT = 2.0
SINKHORN_ITERS = 3

# bf16 carries 8 significand bits; sinkhorn's repeated normalisation widens the
# floor well beyond a single GEMM, so half precision gets a ~100x looser bound.
_TOL = {
    torch.bfloat16: 5e-2,
    torch.float16: 5e-3,
    torch.float32: 5e-4,
}


def _device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    pytest.skip("mHC operators require an accelerator")


def _operands(batch, dtype, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dev = _device()
    return dict(
        residual=torch.randn(batch, HC_MULT, HIDDEN, generator=gen).to(dtype).to(dev),
        fn=(torch.randn((2 + HC_MULT) * HC_MULT, HC_MULT * HIDDEN, generator=gen) * 0.02)
        .to(dtype)
        .to(dev),
        hc_scale=torch.randn(3, generator=gen).to(dev),
        hc_base=(torch.randn((2 + HC_MULT) * HC_MULT, generator=gen) * 0.1).to(dev),
    )


def _pre_args(o):
    return (
        o["residual"],
        o["fn"],
        o["hc_scale"],
        o["hc_base"],
        RMS_EPS,
        SINKHORN_EPS,
        SINKHORN_EPS,
        POST_MULT,
        SINKHORN_ITERS,
    )


@pytest.mark.parametrize("batch", [1, 2, 8, 32, 65])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_mhc_pre_matches_reference(batch, dtype):
    """The dispatched mHC pre must agree with the independent reference."""
    from sglang_fl.dispatch.backends.reference.impl.mhc import mhc_pre_torch

    o = _operands(batch, dtype, seed=batch)
    args = _pre_args(o)
    tol = _TOL[dtype]

    ref_post, ref_comb, ref_layer = mhc_pre_torch(*args)
    got_post, got_comb, got_layer = call_op("mhc_pre", *args)

    for name, got, ref in (
        ("post", got_post, ref_post),
        ("comb", got_comb, ref_comb),
        ("layer_input", got_layer, ref_layer),
    ):
        assert torch.allclose(got.float(), ref.float(), atol=tol, rtol=tol), (
            f"mhc_pre.{name} mismatch at batch={batch} dtype={dtype}: "
            f"max_abs_diff={(got.float() - ref.float()).abs().max().item():.3e} "
            f"(tol={tol})"
        )


@pytest.mark.parametrize("batch", [1, 8, 65])
def test_mhc_pre_sinkhorn_invariants(batch):
    """Sinkhorn row/column sums pin the arithmetic regardless of rounding.

    A tolerance check alone could hide a swapped split or a dropped iteration
    behind a loose bound; these sums match the reference exactly enough that any
    such change shows up.
    """
    from sglang_fl.dispatch.backends.reference.impl.mhc import mhc_pre_torch

    o = _operands(batch, torch.bfloat16, seed=batch)
    args = _pre_args(o)

    _, ref_comb, _ = mhc_pre_torch(*args)
    _, got_comb, _ = call_op("mhc_pre", *args)

    tol = _TOL[torch.bfloat16]
    for axis, label in ((-1, "row"), (-2, "column")):
        got = got_comb.sum(axis).float()
        ref = ref_comb.sum(axis).float()
        assert torch.allclose(got, ref, atol=tol, rtol=tol), (
            f"sinkhorn {label} sums diverged at batch={batch}: "
            f"max_abs_diff={(got - ref).abs().max().item():.3e}"
        )


def test_mhc_post_matches_reference():
    """The dispatched mHC post must agree with the reference on bf16 streams."""
    from sglang_fl.dispatch.backends.reference.impl.mhc import (
        mhc_post_torch,
        mhc_pre_torch,
    )

    dev = _device()
    o = _operands(16, torch.bfloat16, seed=7)
    _, post, comb = mhc_pre_torch(*_pre_args(o))
    x = torch.randn(16, HIDDEN, dtype=torch.bfloat16, device=dev)

    ref = mhc_post_torch(x, o["residual"], post, comb)
    got = call_op("mhc_post", x, o["residual"], post, comb)

    tol = _TOL[torch.bfloat16]
    assert torch.allclose(got.float(), ref.float(), atol=tol, rtol=tol), (
        f"mhc_post mismatch: max_abs_diff={(got.float() - ref.float()).abs().max().item():.3e}"
    )


def test_mhc_post_rejects_non_bf16():
    """The FlagGems kernel stores bf16 unconditionally.

    Accepting a wider dtype would silently narrow the result, so the backend has
    to reject it before any kernel runs rather than return bf16-precision values.
    """
    o = _operands(4, torch.float32, seed=3)
    x = torch.randn(4, HIDDEN, dtype=torch.float32, device=_device())
    post = torch.rand(4, HC_MULT, 1, dtype=torch.float32, device=_device())
    comb = torch.rand(4, HC_MULT, HC_MULT, dtype=torch.float32, device=_device())

    with pytest.raises(NotImplementedError):
        call_op("mhc_post", x, o["residual"], post, comb)


def test_mhc_pre_rejects_unsupported_hc_mult():
    """A different hc_mult must fail loudly, not return a wrongly shaped tensor."""
    o = _operands(2, torch.bfloat16, seed=5)
    o["residual"] = torch.randn(2, 2, HIDDEN, dtype=torch.bfloat16, device=_device())

    with pytest.raises(NotImplementedError):
        call_op("mhc_pre", *_pre_args(o))
