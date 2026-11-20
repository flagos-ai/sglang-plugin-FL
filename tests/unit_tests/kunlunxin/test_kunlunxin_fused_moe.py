# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for ``impl/fused_moe.py``.

The klx expert kernel is stubbed; the wrapper owns the contract that matters:
sglang's ``select_experts`` emits int64 ``topk_ids`` while ``klx_fused_experts``
requires int32, and the kernel is fed ``w13/w2`` plus the activation triple from
``moe_runner_config`` and returns ``StandardCombineInput``.
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

from sglang_fl.dispatch.backends.vendor.kunlunxin.impl.fused_moe import (
    fused_moe_kunlunxin,
)


@pytest.fixture
def klx_fused_experts(monkeypatch):
    record = {}
    namespace = SimpleNamespace()

    def fused_experts(
        hidden_states, w13, w2, topk_weights, topk_ids, inplace, activation, is_gated
    ):
        record.update(
            hidden_states=hidden_states,
            w13=w13,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=inplace,
            activation=activation,
            is_gated=is_gated,
        )
        return hidden_states + 1

    namespace.default = fused_experts
    monkeypatch.setattr(
        torch.ops,
        "sgl_kernel",
        SimpleNamespace(klx_fused_experts=namespace),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "sgl_kernel", types.ModuleType("sgl_kernel"))
    return record


def _moe_args(topk_ids_dtype=torch.int64):
    tokens, hidden, inter, experts = 4, 8, 16, 2
    hidden_states = torch.randn(tokens, hidden)
    topk_weights = torch.rand(tokens, 2)
    topk_ids = torch.zeros(tokens, 2, dtype=topk_ids_dtype)
    layer = SimpleNamespace(
        w13_weight=torch.randn(experts, 2 * inter, hidden),
        w2_weight=torch.randn(experts, hidden, inter),
    )
    obj = SimpleNamespace(
        moe_runner_config=SimpleNamespace(
            activation="silu", inplace=False, is_gated=True
        )
    )
    dispatch_output = SimpleNamespace(
        hidden_states=hidden_states,
        topk_output=SimpleNamespace(
            topk_weights=topk_weights, topk_ids=topk_ids
        ),
    )
    return obj, layer, dispatch_output


def test_int64_topk_ids_are_cast_to_int32(klx_fused_experts):
    obj, layer, dispatch_output = _moe_args(torch.int64)

    fused_moe_kunlunxin(obj, layer, dispatch_output)

    assert klx_fused_experts["topk_ids"].dtype is torch.int32
    assert torch.equal(
        klx_fused_experts["topk_ids"], dispatch_output.topk_output.topk_ids.to(torch.int32)
    )


def test_int32_topk_ids_are_passed_through_untouched(klx_fused_experts):
    obj, layer, dispatch_output = _moe_args(torch.int32)

    fused_moe_kunlunxin(obj, layer, dispatch_output)

    assert klx_fused_experts["topk_ids"] is dispatch_output.topk_output.topk_ids


def test_kernel_receives_sglang_weight_layout_and_runner_config(klx_fused_experts):
    obj, layer, dispatch_output = _moe_args()

    fused_moe_kunlunxin(obj, layer, dispatch_output)

    assert klx_fused_experts["hidden_states"] is dispatch_output.hidden_states
    assert klx_fused_experts["w13"] is layer.w13_weight
    assert klx_fused_experts["w2"] is layer.w2_weight
    assert (
        klx_fused_experts["topk_weights"]
        is dispatch_output.topk_output.topk_weights
    )
    assert klx_fused_experts["activation"] == "silu"
    assert klx_fused_experts["inplace"] is False
    assert klx_fused_experts["is_gated"] is True


def test_result_is_a_standard_combine_input(klx_fused_experts):
    obj, layer, dispatch_output = _moe_args()

    result = fused_moe_kunlunxin(obj, layer, dispatch_output)

    assert isinstance(result, StandardCombineInput)
    assert result.hidden_states.shape == dispatch_output.hidden_states.shape
    assert torch.equal(
        result.hidden_states, dispatch_output.hidden_states + 1
    )
