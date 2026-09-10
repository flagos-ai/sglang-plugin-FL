# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Numerical regression for disabled PDL branches on MUSA Triton 3.2."""

import pytest
import torch


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_rms_norm", [False, True])
@pytest.mark.parametrize("norm_before_gate", [False, True])
@pytest.mark.parametrize("group_size", [128, 256])
def test_gated_layernorm(dtype, is_rms_norm, norm_before_gate, group_size):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA hardware is required")
    layernorm_gated = pytest.importorskip(
        "sglang.kernels.ops.attention.fla.layernorm_gated"
    )
    from sglang_fl.dispatch.backends.vendor.mthreads.triton_compat import (
        patch_triton_pdl_symbols,
    )

    patch_triton_pdl_symbols()
    torch.manual_seed(42)
    x = torch.randn(7, 256, device="musa", dtype=dtype)
    z = torch.randn_like(x)
    weight = torch.randn(256, device="musa", dtype=dtype)
    bias = torch.randn_like(weight) if not is_rms_norm else None
    out = torch.empty_like(x)
    actual, _, _ = layernorm_gated._layer_norm_fwd(
        x,
        weight,
        bias,
        1e-6,
        z=z,
        out=out,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
    )

    # Independent torch reference, with FP32 normalization and gating.
    reference = x.float()
    gate = torch.nn.functional.silu(z.float())
    if not norm_before_gate:
        reference = reference * gate
    grouped = reference.reshape(7, -1, group_size)
    if not is_rms_norm:
        grouped = grouped - grouped.mean(dim=-1, keepdim=True)
    reference = (
        grouped * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + 1e-6)
    ).reshape_as(x)
    reference = reference * weight.float()
    if bias is not None:
        reference = reference + bias.float()
    if norm_before_gate:
        reference = reference * gate
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, reference.to(dtype), rtol=0.01, atol=0.01)
