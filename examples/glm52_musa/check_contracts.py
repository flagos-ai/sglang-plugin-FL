# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate the ATen adapters and RMSNorm bridges on the target MUSA runtime."""

import torch
import torch_musa  # noqa: F401

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.plugins import load_plugins


def main():
    torch.musa.set_device(0)
    torch.manual_seed(101)
    sliced = torch.tensor(
        [[1 << 41, -6], [-(1 << 42), -8]],
        dtype=torch.int64,
        device="musa",
    )[:, :1]
    assert not sliced.is_contiguous()
    reference_min = torch.min(sliced)

    norm_cases = []
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for shape in ((32, 64), (5, 3, 9), (1, 6144)):
            x = torch.randn(shape, dtype=dtype, device="musa")
            normalized = shape[1:]
            weight = torch.randn(normalized, dtype=dtype, device="musa")
            bias = torch.randn_like(weight)
            reference = torch.ops.aten.native_layer_norm(
                x, normalized, weight, bias, 1e-5
            )
            norm_cases.append((x, normalized, weight, bias, reference))
    rms_cases = []
    for width in (512, 2048, 6144):
        x = torch.randn(32, width, dtype=torch.bfloat16, device="musa")
        residual = torch.randn_like(x)
        rms_cases.append((width, x, residual))
    torch.musa.synchronize()

    load_plugins()
    import sglang_fl

    assert sglang_fl.is_plugin_active()
    torch.testing.assert_close(torch.min(sliced), reference_min)
    for x, normalized, weight, bias, reference in norm_cases:
        result = torch.ops.aten.native_layer_norm(x, normalized, weight, bias, 1e-5)
        torch.testing.assert_close(result[0], reference[0], rtol=0.02, atol=0.02)
        for actual, expected in zip(result[1:], reference[1:]):
            torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.001)
    for width, x, residual in rms_cases:
        norm = RMSNorm(width, eps=1e-5).to(device="musa", dtype=torch.bfloat16)
        reference = norm.forward_musa(x.clone())
        torch.testing.assert_close(norm(x.clone()), reference, rtol=0.02, atol=0.02)
        reference = norm.forward_musa(x.clone(), residual.clone())
        lhs, rhs = x.clone(), residual.clone()
        result = norm(lhs, rhs)
        assert result[0] is lhs and result[1] is rhs
        torch.testing.assert_close(result, reference, rtol=0.02, atol=0.02)
    torch.musa.synchronize()
    print("GLM-5.2 MUSA fixed-tensor checks passed")


if __name__ == "__main__":
    main()
