import sys
import types

import pytest

from sglang_fl.dispatch.backends.vendor.ascend.patches import base_fused_op


@pytest.fixture(autouse=True)
def reset_registration(monkeypatch):
    monkeypatch.setattr(base_fused_op, "_registered", False)


def _install_fake_sglang(monkeypatch, calls):
    fused_op = types.ModuleType("sglang.kernels.fused_op")
    unquant = types.ModuleType("sglang.srt.layers.quantization.unquant")

    class FakeBaseFusedOp:
        @classmethod
        def register_oot_forward(cls, op_cls, fn, label):
            calls.append((op_cls, fn, label))

    class FakeUnquantizedFusedMoEMethod:
        @staticmethod
        def forward_npu(*args, **kwargs):
            return None

    fused_op.BaseFusedOp = FakeBaseFusedOp
    unquant.UnquantizedFusedMoEMethod = FakeUnquantizedFusedMoEMethod
    monkeypatch.setitem(sys.modules, fused_op.__name__, fused_op)
    monkeypatch.setitem(sys.modules, unquant.__name__, unquant)
    return FakeUnquantizedFusedMoEMethod


def test_registers_npu_forward_once(monkeypatch):
    calls = []
    method = _install_fake_sglang(monkeypatch, calls)

    base_fused_op.patch_unquantized_fused_moe()
    base_fused_op.patch_unquantized_fused_moe()

    assert calls == [(method, method.forward_npu, "oot")]


def test_missing_registration_api_is_compatible(monkeypatch):
    calls = []
    _install_fake_sglang(monkeypatch, calls)
    sys.modules["sglang.kernels.fused_op"].BaseFusedOp.register_oot_forward = None

    base_fused_op.patch_unquantized_fused_moe()

    assert calls == []
    assert base_fused_op._registered is False
