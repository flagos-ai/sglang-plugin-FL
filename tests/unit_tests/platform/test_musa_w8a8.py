# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU wiring tests; these do not establish MUSA numerical correctness."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = (
    Path(__file__).resolve().parents[3]
    / "sglang_fl/dispatch/backends/vendor"
    / "mthreads/quantization.py"
)
spec = importlib.util.spec_from_file_location("musa_w8a8_test.quantization", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def config(monkeypatch):
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

    upstream = "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors"
    monkeypatch.setitem(
        sys.modules, upstream, SimpleNamespace(CompressedTensorsConfig=Config)
    )
    monkeypatch.setitem(
        sys.modules,
        "musa_w8a8_test.patches",
        SimpleNamespace(require_supported_sglang=lambda: None),
    )
    monkeypatch.setattr(
        module,
        "build_w8a8_schemes",
        lambda: SimpleNamespace(
            linear=lambda *args: ("linear", args), moe=lambda *args: ("moe", args)
        ),
    )
    module.get_musa_compressed_tensors_config.cache_clear()
    cls = module.get_musa_compressed_tensors_config()
    assert module.get_musa_compressed_tensors_config() is cls
    assert Config().get_linear_scheme({}, "a") == ("original-linear", {}, "a")
    yield cls()
    module.get_musa_compressed_tensors_config.cache_clear()


def scheme(dynamic=True, strategy="channel", symmetric=True, quant_type="int"):
    return {
        "weights": SimpleNamespace(strategy=strategy, symmetric=True, type=quant_type),
        "input_activations": SimpleNamespace(
            strategy="token", dynamic=dynamic, symmetric=symmetric, type=quant_type
        ),
    }


@pytest.mark.parametrize(
    "value",
    [
        None,
        scheme(dynamic=False),
        scheme(quant_type="float"),
        {**scheme(), "input_activations": None},
    ],
)
def test_non_dynamic_int8_delegates_to_upstream(config, value):
    linear = {"a": value}
    moe = {f"experts.0.{proj}": value for proj in ("gate_proj", "up_proj", "down_proj")}
    assert config.get_linear_scheme(linear, "a") == ("original-linear", linear, "a")
    assert config.get_moe_scheme(moe, "experts") == ("original-moe", moe, "experts")


def test_dynamic_int8_routes_to_vendor(config):
    value = scheme()
    args = (value["weights"], value["input_activations"])
    assert config.get_linear_scheme({"a": value}, "a") == ("linear", args)
    moe = {f"experts.0.{proj}": value for proj in ("gate_proj", "up_proj", "down_proj")}
    assert config.get_moe_scheme(moe, "experts") == ("moe", args)
    moe["experts.0.down_proj"] = scheme(quant_type="float")
    with pytest.raises(ValueError, match="same quantization"):
        config.get_moe_scheme(moe, "experts")


@pytest.mark.parametrize(
    "value",
    [
        scheme(strategy="tensor"),
        scheme(symmetric=False),
        {**scheme(), "format": "pack-quantized"},
    ],
)
def test_invalid_dynamic_int8_contract_rejected(config, value):
    moe = {f"experts.0.{proj}": value for proj in ("gate_proj", "up_proj", "down_proj")}
    with pytest.raises(NotImplementedError):
        config.get_linear_scheme({"a": value}, "a")
    with pytest.raises(NotImplementedError):
        config.get_moe_scheme(moe, "experts")


def test_factory_failure_propagates(config, monkeypatch):
    def fail():
        raise RuntimeError("required public API is unavailable")

    monkeypatch.setattr(module, "build_w8a8_schemes", fail)
    with pytest.raises(RuntimeError, match="public API"):
        config.get_linear_scheme({"a": scheme()}, "a")
