# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MUSA must replace SGLang 0.5.18's lossy packed GDN decode kernel."""

import pytest
import torch

from sglang_fl.dispatch.backends.vendor.mthreads.runtime_config import (
    install_gdn_packed_decode_backport,
    needs_gdn_packed_decode_backport,
)


def test_installs_fixed_kernel_for_unpatched_sglang_0_5_18():
    class Kernel:
        src = (
            "beta_val = "
            "tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)"
        )

    class KernelModule:
        fused_recurrent_gated_delta_rule_packed_decode_kernel = Kernel()

    fixed_kernel = object()
    assert install_gdn_packed_decode_backport(
        KernelModule, fixed_kernel, "0.5.18"
    )
    assert (
        KernelModule.fused_recurrent_gated_delta_rule_packed_decode_kernel
        is fixed_kernel
    )


def test_keeps_already_fixed_kernel():
    class FixedKernel:
        src = "beta_val = tl.sigmoid(b_val)"

    class KernelModule:
        fused_recurrent_gated_delta_rule_packed_decode_kernel = FixedKernel()

    installed_kernel = KernelModule.fused_recurrent_gated_delta_rule_packed_decode_kernel
    assert not install_gdn_packed_decode_backport(
        KernelModule, object(), "0.5.18"
    )
    assert (
        KernelModule.fused_recurrent_gated_delta_rule_packed_decode_kernel
        is installed_kernel
    )


def test_backport_is_scoped_to_sglang_0_5_18():
    assert needs_gdn_packed_decode_backport("0.5.18")
    assert needs_gdn_packed_decode_backport("0.5.18+musa")
    assert not needs_gdn_packed_decode_backport("0.5.19")


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(),
    reason="requires MUSA hardware",
)
def test_packed_decode_keeps_beta_in_fp32_on_musa():
    import sglang_fl

    sglang_fl.load_plugin()

    from sglang.kernels.ops.attention.fla import fused_recurrent
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import (
        TritonGDNKernel,
    )

    device, dtype = "musa", torch.bfloat16
    mixed_qkv = torch.ones((1, 3), device=device, dtype=dtype)
    a = torch.zeros((1, 1), device=device, dtype=dtype)
    b = torch.full((1, 1), 0.5, device=device, dtype=dtype)
    params = torch.zeros((1,), device=device, dtype=dtype)
    indices = torch.ones((1,), device=device, dtype=torch.int32)
    state = torch.zeros((2, 1, 1, 1), device=device, dtype=torch.float32)
    out = torch.empty((1, 1, 1, 1), device=device, dtype=dtype)

    fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=params,
        dt_bias=params,
        scale=1.0,
        initial_state=state,
        out=out,
        ssm_state_indices=indices,
    )
    torch.musa.synchronize()

    assert TritonGDNKernel.supports_packed_decode is True
    expected = torch.sigmoid(b.float()).reshape(())
    torch.testing.assert_close(state[1, 0, 0, 0], expected, rtol=1e-6, atol=1e-6)
