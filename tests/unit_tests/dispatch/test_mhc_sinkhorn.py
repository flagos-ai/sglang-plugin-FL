# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Numeric and routing tests for the mhc_sinkhorn operator.
#
# Numeric expectations are produced ONLY by the eager torch reference in
# sglang_fl.dispatch.backends.reference.impl.mhc (never by the fused kernel
# under test), per the plugin spec requirement for independent reference
# implementations. The fused Triton kernel requires an accelerator, so its
# numeric cases are gated on torch.npu availability; validation and
# reference-side cases run everywhere.

import pytest
import torch

from sglang_fl.dispatch.backends.reference.impl.mhc import mhc_sinkhorn_torch

try:
    import triton  # noqa: F401
    from sglang_fl.dispatch.backends.flagos.impl.mhc import mhc_sinkhorn_flagos

    _HAS_TRITON = True
except Exception:  # pragma: no cover - triton-less environments
    _HAS_TRITON = False

_SHAPES = [
    (1, 4, 4),
    (8, 4, 4),
    (64, 4, 4),
    (64, 2, 2),
    (3, 3, 3),
    (130, 4, 4),
]

_HAS_NPU = torch.npu.is_available() if hasattr(torch, "npu") else False

requires_npu = pytest.mark.skipif(
    not (_HAS_NPU and _HAS_TRITON), reason="requires NPU + Triton-Ascend"
)


@requires_npu
@pytest.mark.parametrize("shape", _SHAPES)
def test_mhc_sinkhorn_fused_matches_reference(shape):
    torch.manual_seed(0)
    comb = torch.randn(*shape, device="npu", dtype=torch.float32)
    got = mhc_sinkhorn_flagos(comb.clone(), 1e-6, 20)
    ref = mhc_sinkhorn_torch(comb.cpu(), 1e-6, 20).npu()
    torch.testing.assert_close(ref, got, atol=1e-5, rtol=1e-4)


@requires_npu
def test_mhc_sinkhorn_fused_bf16_matches_reference():
    torch.manual_seed(0)
    comb = torch.randn(64, 4, 4, device="npu", dtype=torch.float32).to(torch.bfloat16)
    got = mhc_sinkhorn_flagos(comb.clone(), 1e-6, 20)
    ref = mhc_sinkhorn_torch(comb.cpu(), 1e-6, 20).npu()
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(ref.float(), got.float(), atol=2e-2, rtol=1e-2)


@requires_npu
def test_mhc_sinkhorn_inplace_and_aliasing_contract():
    torch.manual_seed(1)
    comb = torch.randn(64, 4, 4, device="npu", dtype=torch.bfloat16)
    out = mhc_sinkhorn_flagos(comb, 1e-6, 20)
    assert out.data_ptr() == comb.data_ptr()
    sums = out.float().sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-2)


@requires_npu
def test_mhc_sinkhorn_noncontiguous_input_is_copied():
    torch.manual_seed(2)
    base = torch.randn(64, 8, 4, device="npu", dtype=torch.float32)
    view = base[:, ::2, :]  # non-contiguous [64, 4, 4]
    assert not view.is_contiguous()
    out = mhc_sinkhorn_flagos(view, 1e-6, 20)
    ref = mhc_sinkhorn_torch(view.cpu(), 1e-6, 20).npu()
    torch.testing.assert_close(ref, out, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("device", ["cpu"] + (["npu"] if _HAS_NPU else []))
def test_mhc_sinkhorn_reference_row_col_sums(device):
    torch.manual_seed(1)
    comb = torch.randn(16, 4, 4, device=device)
    out = mhc_sinkhorn_torch(comb, 1e-6, 20)
    assert torch.allclose(out.sum(-1), torch.ones_like(out.sum(-1)), atol=1e-4)
    assert torch.allclose(out.sum(-2), torch.ones_like(out.sum(-2)), atol=1e-4)


def test_mhc_sinkhorn_reference_validation_errors():
    with pytest.raises(ValueError):
        mhc_sinkhorn_torch(torch.randn(4, 3), 1e-6, 20)
    with pytest.raises(ValueError):
        mhc_sinkhorn_torch(torch.randn(2, 4, 5), 1e-6, 20)
    with pytest.raises(ValueError):
        mhc_sinkhorn_torch(torch.randn(4, 4), 1e-6, 0)


def test_mhc_sinkhorn_reference_repeat_one():
    torch.manual_seed(2)
    comb = torch.randn(5, 4, 4)
    out = mhc_sinkhorn_torch(comb, 1e-6, 1)
    # single round: softmax row + one column division
    ref = comb.float().softmax(-1) + 1e-6
    ref = ref / (ref.sum(-2, keepdim=True) + 1e-6)
    torch.testing.assert_close(ref.to(comb.dtype), out, atol=1e-5, rtol=1e-4)


@requires_npu
def test_mhc_sinkhorn_registry_routes_vendor_on_ascend():
    """F1: from a real call entry, assert the Ascend backend wins routing."""
    from sglang_fl.dispatch import call_op

    torch.manual_seed(3)
    comb = torch.randn(64, 4, 4, device="npu", dtype=torch.bfloat16)
    out = call_op("mhc_sinkhorn", comb.clone(), 1e-6, 20)
    ref = mhc_sinkhorn_torch(comb.cpu(), 1e-6, 20).npu()
    torch.testing.assert_close(ref.float(), out.float(), atol=2e-2, rtol=1e-2)


@requires_npu
def test_mhc_sinkhorn_policy_reference_forces_reference_impl():
    """Policy override can select the reference implementation explicitly."""
    from sglang_fl.dispatch import call_op
    from sglang_fl.dispatch.policy import with_preference, PREFER_REFERENCE

    torch.manual_seed(4)
    comb = torch.randn(8, 4, 4, device="npu", dtype=torch.bfloat16)
    with with_preference(PREFER_REFERENCE):
        out = call_op("mhc_sinkhorn", comb.clone(), 1e-6, 20)
    ref = mhc_sinkhorn_torch(comb.cpu(), 1e-6, 20).npu()
    torch.testing.assert_close(ref.float(), out.float(), atol=2e-2, rtol=1e-2)
