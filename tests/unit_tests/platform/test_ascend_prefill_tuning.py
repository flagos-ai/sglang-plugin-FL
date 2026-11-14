# Copyright 2026 FlagOS Contributors
"""Prefill patch installation and dispatch contracts; no NPU required."""

import sys
from types import ModuleType, SimpleNamespace as NS
from unittest import mock

import pytest
import torch

from sglang_fl.dispatch.backends.vendor.ascend.patches import prefill_tuning


def _install_modules(monkeypatch, leaves):
    """Mock whole package trees to avoid partially importing/registering SGLang."""
    modules = {}
    for name, attrs in leaves.items():
        parts = name.split(".")
        for end in range(1, len(parts) + 1):
            path = ".".join(parts[:end])
            modules.setdefault(path, ModuleType(path)).__path__ = []
        vars(modules[name]).update(attrs)
    for name, module in modules.items():
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)
        monkeypatch.setitem(sys.modules, name, module)
    return modules


def _tensor(shape):
    return NS(
        ndim=len(shape),
        shape=shape,
        device=NS(type="npu"),
        dtype=torch.bfloat16,
        requires_grad=False,
        is_contiguous=lambda: True,
    )


@pytest.fixture
def nz(monkeypatch):
    calls = []

    class Linear:
        def apply(self, layer, x, bias=None):
            calls.append((layer, x, bias))
            return "STOCK"

    original = Linear.apply
    disabled = mock.Mock(return_value=False)
    capture = mock.Mock(return_value=False)
    npu = dict(get_npu_format=mock.Mock(return_value=2), npu_format_cast=mock.Mock())
    modules = _install_modules(
        monkeypatch,
        {
            "torch_npu": npu,
            "sglang.srt.layers.quantization.unquant": dict(
                UnquantizedLinearMethod=Linear
            ),
            "sglang.srt.environ": dict(
                envs=NS(SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=NS(get=disabled))
            ),
        },
    )
    monkeypatch.setattr(
        torch, "npu", NS(is_current_stream_capturing=capture), raising=False
    )
    monkeypatch.setenv("SGLANG_FL_PREFILL_NZ_GATE_UP", "1")
    return NS(
        cls=Linear,
        original=original,
        calls=calls,
        disabled=disabled,
        capture=capture,
        npu=modules["torch_npu"],
        x=_tensor((16384, 5120)),
        layer=NS(weight=_tensor((17408, 5120))),
    )


def test_default_off(monkeypatch, nz):
    for key in ("SGLANG_FL_PREFILL_NZ_GATE_UP", "SGLANG_FL_GDN_GATING_BLOCK24"):
        monkeypatch.delenv(key, raising=False)
    prefill_tuning.patch_prefill_tuning()
    assert nz.cls.apply is nz.original


@pytest.mark.parametrize("guard", ["disabled", "format", "capture"])
def test_nz_fallback(nz, guard):
    nz.disabled.return_value = guard == "disabled"
    nz.npu.get_npu_format.return_value = 0 if guard == "format" else 2
    nz.capture.return_value = guard == "capture"
    assert prefill_tuning._patch_ephemeral_nz() is (guard != "disabled")
    if guard == "disabled":
        assert nz.cls.apply is nz.original
    assert nz.cls().apply(nz.layer, nz.x) == "STOCK"
    assert nz.calls == [(nz.layer, nz.x, None)]
    nz.npu.npu_format_cast.assert_not_called()


def test_nz_signature_guard(nz):
    def wrong(self, x):
        pass

    nz.cls.apply = wrong
    assert not prefill_tuning._patch_ephemeral_nz()
    assert nz.cls.apply is wrong


def test_nz_dispatch_and_idempotence(monkeypatch, nz):
    linear = mock.Mock(return_value="NZ")
    monkeypatch.setattr(torch.nn.functional, "linear", linear)
    assert prefill_tuning._patch_ephemeral_nz()
    wrapped = nz.cls.apply
    assert not prefill_tuning._patch_ephemeral_nz()
    assert nz.cls.apply is wrapped
    weight = nz.layer.weight
    assert nz.cls().apply(nz.layer, nz.x) == "NZ"
    nz.npu.npu_format_cast.assert_called_once_with(weight, 29)
    linear.assert_called_once_with(nz.x, nz.npu.npu_format_cast.return_value)
    assert nz.layer.weight is weight and not nz.calls


def _kernel_abi(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    batch,
    seq_len,
    NUM_HEADS,
    beta,
    threshold,
    BLK_HEADS,
):
    pass


@pytest.fixture
def gating(monkeypatch):
    calls = []

    def original(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
        calls.append((A_log, a, b, dt_bias, beta, threshold))
        return "STOCK", "BETA"

    kernel = mock.MagicMock(fn=_kernel_abi)
    name = "sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend"
    modules = _install_modules(
        monkeypatch,
        {
            "triton.runtime": dict(driver=NS()),
            "sgl_kernel_npu.fla.fused_gdn_gating": dict(fused_gdn_gating_kernel=kernel),
            name: dict(fused_gdn_gating=original),
        },
    )
    monkeypatch.setenv("SGLANG_FL_GDN_GATING_BLOCK24", "1")
    return NS(kernel=kernel, backend=modules[name], original=original, calls=calls)


@pytest.mark.parametrize("fn", [None, lambda a, b: None])
def test_gating_abi_guard(gating, fn):
    gating.kernel.fn = fn
    assert not prefill_tuning._patch_gating()
    assert gating.backend.fused_gdn_gating is gating.original


def test_gating_idempotence_and_fallback(gating):
    assert prefill_tuning._patch_gating()
    wrapped = gating.backend.fused_gdn_gating
    assert not prefill_tuning._patch_gating()
    assert gating.backend.fused_gdn_gating is wrapped
    args = (
        torch.zeros(24),
        torch.zeros(16384, 24, dtype=torch.bfloat16),
        torch.zeros(16384, 24, dtype=torch.bfloat16),
        torch.zeros(24),
    )
    for tensors in (args, (args[0], args[1][:64], args[2][:64], args[3])):
        assert wrapped(*tensors) == ("STOCK", "BETA")
    assert len(gating.calls) == 2
    gating.kernel.__getitem__.assert_not_called()
