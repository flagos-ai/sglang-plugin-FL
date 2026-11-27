# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from collections import namedtuple
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

_MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "sglang_fl"
    / "dispatch"
    / "backends"
    / "vendor"
    / "mthreads"
    / "impl"
    / "topk.py"
)
_SPEC = importlib.util.spec_from_file_location("test_musa_topk_impl", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOPK_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TOPK_MODULE)
topk_musa = _TOPK_MODULE.topk_musa


def _topk_output_type():
    return namedtuple("StandardTopKOutput", "topk_weights topk_ids router_logits")


def test_sigmoid_routed_scaling_uses_sglang_generic_path(monkeypatch):
    calls = []
    topk_module = ModuleType("sglang.srt.layers.moe.topk")
    topk_module.StandardTopKOutput = _topk_output_type()
    expected = object()

    def select_experts(**kwargs):
        calls.append(kwargs)
        return expected

    topk_module.select_experts = select_experts
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.moe.topk", topk_module)

    config = SimpleNamespace(
        scoring_func="sigmoid",
        apply_routed_scaling_factor_on_output=True,
        routed_scaling_factor=2.5,
        torch_native=False,
    )
    hidden_states = torch.zeros(3, 4)
    router_logits = torch.zeros(3, 8)
    result = topk_musa(
        SimpleNamespace(topk_config=config, layer_id=7),
        hidden_states,
        router_logits,
    )

    assert result is expected
    assert len(calls) == 1
    assert calls[0]["topk_config"] is config
    assert calls[0]["hidden_states"] is hidden_states
    assert calls[0]["router_logits"] is router_logits
    assert calls[0]["layer_id"] == 7
    assert not config.torch_native


@pytest.mark.parametrize("scoring_func", ["sigmoid", "softmax"])
def test_vendor_path_is_used_when_contract_is_supported(monkeypatch, scoring_func):
    topk_module = ModuleType("sglang.srt.layers.moe.topk")
    topk_module.StandardTopKOutput = _topk_output_type()

    def unexpected_select_experts(**kwargs):
        raise AssertionError("generic path should not be selected")

    topk_module.select_experts = unexpected_select_experts
    musa_topk_module = ModuleType("sglang.srt.hardware_backend.musa.kernels.topk")

    def topk_sigmoid(weights, ids, gating, renormalize, correction_bias):
        weights.fill_(0.25)
        ids.zero_()

    def topk_softmax(
        weights, ids, gating, renormalize, moe_softcapping, correction_bias
    ):
        weights.fill_(0.25)
        ids.zero_()

    musa_topk_module.topk_sigmoid = topk_sigmoid
    musa_topk_module.topk_softmax = topk_softmax
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.moe.topk", topk_module)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.hardware_backend.musa.kernels.topk",
        musa_topk_module,
    )

    config = SimpleNamespace(
        use_grouped_topk=False,
        custom_routing_function=None,
        num_fused_shared_experts=0,
        top_k=2,
        renormalize=True,
        scoring_func=scoring_func,
        correction_bias=None,
        apply_routed_scaling_factor_on_output=False,
        routed_scaling_factor=2.5,
    )
    output = topk_musa(
        SimpleNamespace(topk_config=config, layer_id=0),
        torch.zeros(3, 4),
        torch.zeros(3, 8),
    )

    torch.testing.assert_close(output.topk_weights, torch.full((3, 2), 0.25))
