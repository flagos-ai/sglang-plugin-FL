# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""HCU delegates Triton MoE weight-layout metadata to SGLang when available."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def fused_moe():
    path = (
        Path(__file__).resolve().parents[3]
        / "sglang_fl/dispatch/backends/vendor/hygon/impl/fused_moe.py"
    )
    spec = importlib.util.spec_from_file_location("hcu_fused_moe_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fused_moe_hcu


def install_quant_info_stub(monkeypatch, constructor):
    name = "sglang.srt.layers.moe.moe_runner.triton"
    module = ModuleType(name)
    module.TritonMoeQuantInfo = constructor
    monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize("interleaved", [False, True])
def test_new_getter_preserves_complete_quant_info(fused_moe, monkeypatch, interleaved):
    # The getter may carry additional upstream metadata unknown to this bridge.
    quant_info = SimpleNamespace(
        fuse_swiglu_interleaved=interleaved,
        upstream_metadata=object(),
    )
    constructor = Mock(side_effect=AssertionError("must not rebuild quant_info"))
    install_quant_info_stub(monkeypatch, constructor)
    get_quant_info = Mock(return_value=quant_info)
    output = object()
    run = Mock(return_value=output)
    obj = SimpleNamespace(
        get_triton_quant_info=get_quant_info, runner=SimpleNamespace(run=run)
    )
    layer, dispatch_output = object(), object()

    assert fused_moe(obj, layer, dispatch_output) is output
    get_quant_info.assert_called_once_with(layer)
    run.assert_called_once_with(dispatch_output, quant_info)
    assert run.call_args.args[1] is quant_info
    assert run.call_args.args[1].fuse_swiglu_interleaved is interleaved
    constructor.assert_not_called()


@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("getter_missing", [False, True])
def test_legacy_fallback_keeps_weights_and_optional_biases(
    fused_moe, monkeypatch, with_bias, getter_missing
):
    constructor = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    install_quant_info_stub(monkeypatch, constructor)
    layer = SimpleNamespace(w13_weight=object(), w2_weight=object())
    if with_bias:
        layer.w13_weight_bias, layer.w2_weight_bias = object(), object()
    output, dispatch_output = object(), object()
    run = Mock(return_value=output)
    obj = SimpleNamespace(runner=SimpleNamespace(run=run))
    if not getter_missing:
        obj.get_triton_quant_info = None

    assert fused_moe(obj, layer, dispatch_output) is output
    expected = dict(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        b13=getattr(layer, "w13_weight_bias", None),
        b2=getattr(layer, "w2_weight_bias", None),
    )
    constructor.assert_called_once_with(**expected)
    quant_info = run.call_args.args[1]
    assert vars(quant_info) == expected
    run.assert_called_once_with(dispatch_output, quant_info)


def test_getter_failure_is_propagated_without_legacy_fallback(fused_moe, monkeypatch):
    constructor = Mock()
    install_quant_info_stub(monkeypatch, constructor)
    run = Mock()
    obj = SimpleNamespace(
        get_triton_quant_info=Mock(side_effect=RuntimeError("invalid upstream layout")),
        runner=SimpleNamespace(run=run),
    )
    with pytest.raises(RuntimeError, match="invalid upstream layout"):
        fused_moe(obj, object(), object())
    constructor.assert_not_called()
    run.assert_not_called()
