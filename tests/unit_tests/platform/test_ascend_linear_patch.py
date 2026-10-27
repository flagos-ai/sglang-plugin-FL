# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for the Ascend unquantized linear dispatch boundary."""

import importlib.util
import inspect
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


_PATCH_FILE = (
    Path(__file__).resolve().parents[3]
    / "sglang_fl/dispatch/backends/vendor/ascend/patches/linear.py"
)
_NPU_DEVICE = SimpleNamespace(type="npu", index=0)


class _NpuCpuTensor(torch.Tensor):
    """Exercise the NPU guard using actual CPU storage and arithmetic."""

    @property
    def device(self):
        return _NPU_DEVICE


def _npu(value):
    return value.as_subclass(_NpuCpuTensor)


@pytest.fixture
def patch_module():
    spec = importlib.util.spec_from_file_location("ascend_linear_contract", _PATCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_upstream(monkeypatch, original=None):
    calls = []

    class Method:
        def apply(self, layer, x, bias=None):
            calls.append((self, layer, x, bias))
            return "original"

    if original is not None:
        Method.apply = original
    modules = {}
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.unquant",
    ):
        modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
    for name, module in modules.items():
        parent, _, child = name.rpartition(".")
        if parent in modules:
            setattr(modules[parent], child, module)
    modules["sglang.srt.layers.quantization.unquant"].UnquantizedLinearMethod = Method
    return Method, calls


@pytest.mark.parametrize("flag_env", ("0", "1"))
def test_selected_case_uses_normal_mm_and_preserves_bf16_values(
    monkeypatch, patch_module, flag_env
):
    monkeypatch.setenv("USE_FLAGGEMS", flag_env)
    method, delegated = _install_upstream(monkeypatch)
    patch_module.patch_unquantized_linear_mm()
    x = _npu(torch.arange(6).reshape(2, 3).to(torch.bfloat16) / 4)
    weight = _npu(torch.arange(12).reshape(4, 3).to(torch.bfloat16) / 32)
    expected = torch.nn.functional.linear(
        x.as_subclass(torch.Tensor), weight.as_subclass(torch.Tensor)
    )
    calls = []
    original_mm = torch.mm

    def configured_mm(left, right):
        calls.append((left, right))
        return original_mm(left, right)

    monkeypatch.setattr(patch_module.torch, "mm", configured_mm)
    output = method().apply(SimpleNamespace(weight=weight), x)
    assert not delegated and len(calls) == 1
    assert calls[0][0] is x
    assert (
        calls[0][1].untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
    )
    assert calls[0][1].stride() == (1, 3)
    assert output.shape == (2, 4) and output.dtype == torch.bfloat16
    torch.testing.assert_close(
        output.as_subclass(torch.Tensor), expected, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "case", ["cpu", "cuda", "rank1", "rank3", "bias", "fp16", "fp32", "weight_fp32"]
)
def test_other_contracts_delegate_original_arguments(monkeypatch, patch_module, case):
    method, calls = _install_upstream(monkeypatch)
    patch_module.patch_unquantized_linear_mm()
    x = _npu(torch.ones((2, 3), dtype=torch.bfloat16))
    weight = _npu(torch.ones((4, 3), dtype=torch.bfloat16))
    bias = None
    if case == "cpu":
        x = x.as_subclass(torch.Tensor)
    elif case == "cuda":
        x = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    elif case == "rank1":
        x = x[0]
    elif case == "rank3":
        x = x.unsqueeze(0)
    elif case == "bias":
        bias = object()
    elif case == "fp16":
        x = x.to(torch.float16)
    elif case == "fp32":
        x = x.float()
    else:
        weight = weight.float()
    layer = SimpleNamespace(weight=weight)
    instance = method()
    assert instance.apply(layer, x, bias=bias) == "original"
    assert len(calls) == 1
    assert all(
        actual is expected
        for actual, expected in zip(calls[0], (instance, layer, x, bias))
    )


def test_empty_matrix_keeps_original_and_never_launches_mm(monkeypatch, patch_module):
    delegated = []

    def original(self, layer, x, bias=None):
        delegated.append((layer, x, bias))
        return torch.nn.functional.linear(
            x.as_subclass(torch.Tensor), layer.weight.as_subclass(torch.Tensor), bias
        )

    method, _ = _install_upstream(monkeypatch, original)
    patch_module.patch_unquantized_linear_mm()

    def rejected_mm(*args, **kwargs):
        raise AssertionError("Empty NPU input must not launch the FlagGems mm kernel")

    monkeypatch.setattr(patch_module.torch, "mm", rejected_mm)
    x = _npu(torch.empty((0, 3), dtype=torch.bfloat16))
    weight = _npu(torch.ones((4, 3), dtype=torch.bfloat16))
    layer = SimpleNamespace(weight=weight)
    output = method().apply(layer, x)
    assert output.shape == (0, 4) and output.dtype == torch.bfloat16
    assert len(delegated) == 1 and delegated[0][0] is layer and delegated[0][1] is x


def test_wrapper_does_not_detach_autograd(monkeypatch, patch_module):
    method, delegated = _install_upstream(monkeypatch)
    patch_module.patch_unquantized_linear_mm()
    base_x = torch.ones((2, 3), dtype=torch.bfloat16, requires_grad=True)
    base_weight = torch.ones((4, 3), dtype=torch.bfloat16, requires_grad=True)
    output = method().apply(SimpleNamespace(weight=_npu(base_weight)), _npu(base_x))
    output.as_subclass(torch.Tensor).float().sum().backward()
    assert not delegated
    torch.testing.assert_close(base_x.grad, torch.full_like(base_x, 4), rtol=0, atol=0)
    torch.testing.assert_close(
        base_weight.grad, torch.full_like(base_weight, 2), rtol=0, atol=0
    )


@pytest.mark.parametrize("case", ["rank", "device", "dimensions"])
def test_invalid_selected_weight_fails_closed(monkeypatch, patch_module, case):
    method, delegated = _install_upstream(monkeypatch)
    patch_module.patch_unquantized_linear_mm()
    x = _npu(torch.ones((2, 3), dtype=torch.bfloat16))
    weight = _npu(torch.ones((4, 3), dtype=torch.bfloat16))
    if case == "rank":
        weight = weight[0]
    elif case == "device":
        weight = weight.as_subclass(torch.Tensor)
    else:
        weight = weight[:, :2]
    with pytest.raises(RuntimeError, match="shape/device contract"):
        method().apply(SimpleNamespace(weight=weight), x)
    assert not delegated


def test_incompatible_output_does_not_silently_fallback(monkeypatch, patch_module):
    method, delegated = _install_upstream(monkeypatch)
    patch_module.patch_unquantized_linear_mm()
    monkeypatch.setattr(patch_module.torch, "mm", lambda x, weight: x)
    x = _npu(torch.ones((2, 3), dtype=torch.bfloat16))
    weight = _npu(torch.ones((4, 3), dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="output contract"):
        method().apply(SimpleNamespace(weight=weight), x)
    assert not delegated


@pytest.mark.parametrize("case", ["rename", "required_bias", "keyword_only", "default"])
def test_upstream_interface_change_rejected_before_mutation(
    monkeypatch, patch_module, case
):
    if case == "rename":

        def original(self, layer, renamed, bias=None):
            pass
    elif case == "required_bias":

        def original(self, layer, x, bias):
            pass
    elif case == "keyword_only":

        def original(self, layer, x, *, bias=None):
            pass
    else:

        def original(self, layer, x, bias=0):
            pass

    method, _ = _install_upstream(monkeypatch, original)
    with pytest.raises(RuntimeError, match="Unsupported SGLang interface"):
        patch_module.patch_unquantized_linear_mm()
    assert method.apply is original


def test_patch_is_idempotent_and_preserves_signature(monkeypatch, patch_module):
    method, _ = _install_upstream(monkeypatch)
    original = method.apply
    patch_module.patch_unquantized_linear_mm()
    installed = method.apply
    patch_module.patch_unquantized_linear_mm()
    assert method.apply is installed and installed is not original
    assert installed.__wrapped__ is original
    assert inspect.signature(installed) == inspect.signature(original)
