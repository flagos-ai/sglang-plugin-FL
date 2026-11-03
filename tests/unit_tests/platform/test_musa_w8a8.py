# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU wiring tests; these do not establish MUSA numerical correctness."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = (
    Path(__file__).resolve().parents[3]
    / "sglang_fl/dispatch/backends/vendor"
    / "mthreads/patches/w8a8.py"
)
spec = importlib.util.spec_from_file_location("musa_w8a8_patch_test", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    class Config:
        sparsity_scheme_map = {}

        def get_linear_scheme(self, layer, layer_name=None):
            return ("original-linear", layer, layer_name)

        def get_moe_scheme(self, layer, layer_name=None):
            return ("original-moe", layer, layer_name)

        def get_scheme_dict(self, layer, layer_name=None):
            return layer.get(layer_name)

        def _is_dynamic_token_w8a8(self, weight_quant, input_quant):
            return bool(weight_quant and input_quant and input_quant.dynamic)

        def _add_fused_moe_to_target_scheme_map(self):
            pass

    return SimpleNamespace(CompressedTensorsConfig=Config)


def scheme(dynamic=True, strategy="channel", symmetric=True):
    return {
        "weights": SimpleNamespace(strategy=strategy, symmetric=symmetric, type="int"),
        "input_activations": SimpleNamespace(
            strategy="token", dynamic=dynamic, symmetric=symmetric, type="int"
        ),
    }


def test_fp8_and_weight_only_schemes_do_not_enter_int8_factory():
    upstream = fixture()
    module.install_w8a8_patches(upstream, lambda: True, lambda: None)
    value = scheme()
    value["weights"].type = "float"
    value["input_activations"].type = "float"
    layer = {"a": value}
    assert (
        upstream.CompressedTensorsConfig().get_linear_scheme(layer, "a")[0]
        == "original-linear"
    )
    value["input_activations"] = None
    assert (
        upstream.CompressedTensorsConfig().get_linear_scheme(layer, "a")[0]
        == "original-linear"
    )


def test_platform_scope_and_idempotence():
    upstream = fixture()
    musa = [True]
    factories = SimpleNamespace(
        linear=lambda *args: ("linear", args), moe=lambda *args: ("moe", args)
    )
    assert module.install_w8a8_patches(upstream, lambda: musa[0], lambda: factories)
    method = upstream.CompressedTensorsConfig.get_linear_scheme
    assert not module.install_w8a8_patches(upstream, lambda: True, lambda: factories)
    assert method is upstream.CompressedTensorsConfig.get_linear_scheme
    cfg = upstream.CompressedTensorsConfig()
    layer = {"a": scheme()}
    assert cfg.get_linear_scheme(layer, "a")[0] == "linear"
    musa[0] = False
    assert cfg.get_linear_scheme(layer, "a") == ("original-linear", layer, "a")


def test_moe_requires_consistent_projections_and_preserves_ignored_layers():
    upstream = fixture()
    factories = SimpleNamespace(linear=lambda *args: "linear", moe=lambda *args: "moe")
    module.install_w8a8_patches(upstream, lambda: True, lambda: factories)
    cfg = upstream.CompressedTensorsConfig()
    value = scheme()
    layer = {
        f"experts.0.{proj}": value for proj in ("gate_proj", "up_proj", "down_proj")
    }
    assert cfg.get_moe_scheme(layer, "experts") == "moe"
    layer["experts.0.down_proj"] = scheme(symmetric=False)
    with pytest.raises(ValueError, match="same quantization"):
        cfg.get_moe_scheme(layer, "experts")
    assert cfg.get_moe_scheme({}, "experts") == ("original-moe", {}, "experts")


@pytest.mark.parametrize("value", [scheme(strategy="tensor"), scheme(symmetric=False)])
def test_unsupported_contract_rejected_before_factory(value):
    upstream = fixture()

    def forbidden():
        raise AssertionError("Factory must not execute for an unsupported contract")

    module.install_w8a8_patches(upstream, lambda: True, forbidden)
    with pytest.raises(NotImplementedError, match="symmetric.*channel"):
        upstream.CompressedTensorsConfig().get_linear_scheme({"a": value}, "a")


def test_missing_capability_and_signature_drift_do_not_modify_methods():
    upstream = fixture()
    cls = upstream.CompressedTensorsConfig
    original = cls.get_linear_scheme
    del cls.get_scheme_dict
    with pytest.raises(RuntimeError, match="get_scheme_dict"):
        module.install_w8a8_patches(upstream, lambda: True, lambda: None)
    assert cls.get_linear_scheme is original
    upstream = fixture()
    original = upstream.CompressedTensorsConfig.get_linear_scheme
    upstream.CompressedTensorsConfig.get_moe_scheme = lambda self, layer, renamed: None
    with pytest.raises(RuntimeError, match="layer_name"):
        module.install_w8a8_patches(upstream, lambda: True, lambda: None)
    assert upstream.CompressedTensorsConfig.get_linear_scheme is original


def test_factory_failure_does_not_fall_back_or_poison_future_calls():
    upstream = fixture()
    calls = []

    def failing():
        calls.append(True)
        raise RuntimeError("required public API is unavailable")

    module.install_w8a8_patches(upstream, lambda: True, failing)
    cfg = upstream.CompressedTensorsConfig()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="public API"):
            cfg.get_linear_scheme({"a": scheme()}, "a")
    assert len(calls) == 2


def test_non_dynamic_scheme_retains_original_path():
    upstream = fixture()
    module.install_w8a8_patches(upstream, lambda: True, lambda: None)
    layer = {"a": scheme(dynamic=False)}
    assert upstream.CompressedTensorsConfig().get_linear_scheme(layer, "a") == (
        "original-linear",
        layer,
        "a",
    )


def test_upgrade_with_new_parameters_is_not_silently_accepted():
    upstream = fixture()
    cls = upstream.CompressedTensorsConfig
    original = cls.get_linear_scheme
    cls.get_moe_scheme = lambda self, layer, layer_name=None, extra=False: None
    with pytest.raises(RuntimeError, match="unsupported signature"):
        module.install_w8a8_patches(upstream, lambda: True, lambda: None)
    assert cls.get_linear_scheme is original
