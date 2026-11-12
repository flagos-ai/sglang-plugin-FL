# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the Kunlunxin dispatch backend and its registrations.

Covers:
  * ``vendor/kunlunxin/kunlunxin.py``   — availability probe + op delegation
  * ``vendor/kunlunxin/register_ops.py`` — VENDOR OpImpl registration/gating
  * ``vendor/kunlunxin/register_platform.py`` — OOT attention-backend factory

The vendor kernel packages are unavailable off-device, so the probe is stubbed
through ``sys.modules`` and the ``impl.*`` modules are replaced with recorders.
"""

import sys
import types
from types import SimpleNamespace

import pytest

from sglang_fl.dispatch.backends.vendor.kunlunxin import (
    kunlunxin as kunlunxin_module,
    register_ops,
    register_platform,
)
from sglang_fl.dispatch.backends.vendor.kunlunxin.impl import (
    activation as activation_impl,
    fla as fla_impl,
    fused_moe as fused_moe_impl,
)
from sglang_fl.dispatch.registry import OpRegistry
from sglang_fl.dispatch.types import BackendImplKind, BackendPriority

KunlunxinBackend = kunlunxin_module.KunlunxinBackend

_EXPECTED_OPS = (
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule_packed_decode",
    "fused_moe",
)


@pytest.fixture(autouse=True)
def _isolate_availability(monkeypatch):
    """Do not leak the cached availability probe between tests/processes."""
    monkeypatch.setattr(KunlunxinBackend, "_available", None, raising=False)


def _install_torch_xmlir(monkeypatch, *, available=True, version="KL3", calls=None):
    module = types.ModuleType("torch_xmlir")
    module.xpu = SimpleNamespace(is_available=lambda: available)

    def get_xpu_version():
        if calls is not None:
            calls.append(version)
        return version

    module.get_xpu_version = get_xpu_version
    monkeypatch.setitem(sys.modules, "torch_xmlir", module)
    return module


def test_backend_identity():
    backend = KunlunxinBackend()
    assert backend.name == "kunlunxin"
    assert backend.vendor == "kunlunxin"


def test_is_available_accepts_kl3_only(monkeypatch):
    _install_torch_xmlir(monkeypatch, version="KL3")
    assert KunlunxinBackend().is_available() is True


@pytest.mark.parametrize("version", ["KL2", "KL1", "", None])
def test_is_available_rejects_other_generations(monkeypatch, version):
    _install_torch_xmlir(monkeypatch, version=version)
    assert KunlunxinBackend().is_available() is False


def test_is_available_false_when_no_xpu(monkeypatch):
    _install_torch_xmlir(monkeypatch, available=False)
    assert KunlunxinBackend().is_available() is False


def test_is_available_false_without_torch_xmlir(monkeypatch):
    # ``None`` in sys.modules makes ``import torch_xmlir`` raise ImportError,
    # which is how the probe behaves on every non-Kunlun host.
    monkeypatch.setitem(sys.modules, "torch_xmlir", None)
    assert KunlunxinBackend().is_available() is False


def test_is_available_is_cached(monkeypatch):
    calls = []
    _install_torch_xmlir(monkeypatch, calls=calls)
    backend = KunlunxinBackend()
    assert backend.is_available() is True
    assert backend.is_available() is True
    assert len(calls) == 1


def test_chunk_gated_delta_rule_delegates_in_declared_order(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        fla_impl,
        "chunk_gated_delta_rule_kunlunxin",
        lambda *args: seen.setdefault("args", args),
    )
    args = tuple(
        [f"arg{i}" for i in range(6)]
        + ["initial_state", "initial_state_indices", "cu_seqlens", True, True]
    )
    assert KunlunxinBackend().chunk_gated_delta_rule(*args) == args
    assert seen["args"] == args


def test_packed_decode_delegates_in_declared_order(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        fla_impl,
        "fused_recurrent_gated_delta_rule_packed_decode_kunlunxin",
        lambda *args: seen.setdefault("args", args),
    )
    args = tuple(f"arg{i}" for i in range(9)) + (True,)
    assert KunlunxinBackend().fused_recurrent_gated_delta_rule_packed_decode(*args) == args
    assert seen["args"] == args


def test_fused_moe_delegates(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        fused_moe_impl,
        "fused_moe_kunlunxin",
        lambda *args: seen.setdefault("args", args),
    )
    obj, layer, dispatch_output = object(), object(), object()
    assert KunlunxinBackend().fused_moe(obj, layer, dispatch_output) == (
        obj,
        layer,
        dispatch_output,
    )
    assert seen["args"] == (obj, layer, dispatch_output)


def test_register_ops_registers_vendor_impls():
    registry = OpRegistry()
    register_ops.register_builtins(registry)

    for op_name in _EXPECTED_OPS:
        impl = registry.get_implementation(op_name, "vendor.kunlunxin")
        assert impl is not None, op_name
        assert impl.kind is BackendImplKind.VENDOR
        assert impl.vendor == "kunlunxin"
        assert impl.priority == BackendPriority.VENDOR
        # ``is_available`` is bound as the wrapped function's gate, so the
        # dispatcher can skip the vendor impl on other hardware.
        assert impl.fn._is_available.__func__ is KunlunxinBackend.is_available


def test_registered_wrappers_keep_the_backend_method_names():
    registry = OpRegistry()
    register_ops.register_builtins(registry)

    for op_name in _EXPECTED_OPS:
        impl = registry.get_implementation(op_name, "vendor.kunlunxin")
        assert impl.fn.__name__ == op_name


def test_register_ops_gate_follows_hardware(monkeypatch):
    registry = OpRegistry()
    register_ops.register_builtins(registry)
    impl = registry.get_implementation("fused_moe", "vendor.kunlunxin")

    monkeypatch.setitem(sys.modules, "torch_xmlir", None)
    assert impl.fn._is_available() is False

    _install_torch_xmlir(monkeypatch, version="KL3")
    monkeypatch.setattr(KunlunxinBackend, "_available", None, raising=False)
    assert impl.fn._is_available() is True


def test_register_platform_registers_attention_backend_factory(monkeypatch):
    from sglang.srt.layers.attention import attention_registry

    from sglang_fl.dispatch.backends.vendor.kunlunxin.impl import (
        attention_backend as attention_impl,
    )

    created = []

    class RecordingBackend:
        def __init__(self, runner):
            created.append(runner)

    monkeypatch.setattr(attention_impl, "KunlunxinBackend", RecordingBackend)
    factory = attention_registry.ATTENTION_BACKENDS["kunlunxin"]
    runner = SimpleNamespace(device="fake")
    assert isinstance(factory(runner), RecordingBackend)
    assert created == [runner]


def test_register_platform_module_is_the_registration_seam():
    """``platform.PlatformFL.init_backend`` imports this exact module name."""
    assert register_platform.__name__.endswith(".kunlunxin.register_platform")
