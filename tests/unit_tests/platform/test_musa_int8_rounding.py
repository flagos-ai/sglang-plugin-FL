# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Test launch routing only; numerical boundary cases require real MUSA."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = (
    Path(__file__).resolve().parents[3]
    / "sglang_fl/dispatch/backends/vendor"
    / "mthreads/patches/int8_rounding.py"
)
spec = importlib.util.spec_from_file_location("musa_int8_rounding_test", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Kernel:
    def __init__(self, calls):
        self.calls = calls

    @staticmethod
    def fn(
        x_ptr,
        xq_ptr,
        scale_ptr,
        x_sum_ptr,
        stride_x,
        stride_xq,
        N,
        CAL_SUM,
        BLOCK,
        IS_HIP,
    ):
        pass

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
            return "launched"

        return launch


def test_only_musa_selects_generic_rounding_and_install_is_idempotent():
    calls = []
    original = Kernel(calls)
    upstream = SimpleNamespace(_per_token_quant_int8=original)
    assert module.install_int8_rounding_patch(upstream)
    installed = upstream._per_token_quant_int8
    assert not module.install_int8_rounding_patch(upstream)
    assert installed is upstream._per_token_quant_int8
    for device in ("musa", "cuda", "cpu"):
        tensor = SimpleNamespace(device=SimpleNamespace(type=device))
        kwargs = {"IS_HIP": False, "CAL_SUM": True, "BLOCK": 512}
        assert installed[(4,)](tensor, "q", "scales", "sum", **kwargs) == "launched"
        assert calls[-1][2]["IS_HIP"] is (device == "musa")
        assert calls[-1][2]["CAL_SUM"] is True
        assert kwargs["IS_HIP"] is False
    assert installed.fn is original.fn


def test_execution_error_is_propagated_without_second_launch():
    calls = []

    class FailingKernel(Kernel):
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append(True)
                raise RuntimeError("device execution failed")

            return launch

    upstream = SimpleNamespace(_per_token_quant_int8=FailingKernel(calls))
    module.install_int8_rounding_patch(upstream)
    with pytest.raises(RuntimeError, match="device execution"):
        upstream._per_token_quant_int8[(1,)](
            SimpleNamespace(device=SimpleNamespace(type="musa")), IS_HIP=False
        )
    assert len(calls) == 1


def test_signature_drift_is_rejected_before_mutation():
    original = SimpleNamespace(fn=lambda x: x)
    upstream = SimpleNamespace(_per_token_quant_int8=original)
    with pytest.raises(RuntimeError, match="IS_HIP"):
        module.install_int8_rounding_patch(upstream)
    assert upstream._per_token_quant_int8 is original
