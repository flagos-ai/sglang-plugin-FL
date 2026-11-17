# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TXDA vendor fused_moe forwards full MoE semantics and rejects unsupported configs (CPU)."""

import sys
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("sglang.srt.layers.moe.token_dispatcher.standard")

from sglang_fl.dispatch.backends.vendor.tsingmicro.impl import (  # noqa: E402
    fused_moe as fm,
)

E, N, K, I, M, TOPK = 4, 8, 8, 4, 3, 2  # experts, hidden, hidden, intermediate, tokens


def _setup(monkeypatch, *, scaling=None, num_local=E, activation="silu"):
    calls = {}

    def fake_impl(**kwargs):
        calls.update(kwargs)
        return torch.ones(M, N)

    monkeypatch.setitem(sys.modules, "flag_gems", SimpleNamespace(fused_experts_impl=fake_impl))
    config = SimpleNamespace(
        activation=activation,
        is_gated=True,
        no_combine=False,
        apply_router_weight_on_input=False,
        num_experts=E,
        num_local_experts=num_local,
        routed_scaling_factor=scaling,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        swiglu_limit=None,
    )
    obj = SimpleNamespace(moe_runner_config=config)
    layer = SimpleNamespace(
        w13_weight=torch.zeros(E, 2 * I, K, dtype=torch.bfloat16),
        w2_weight=torch.zeros(E, N, I, dtype=torch.bfloat16),
    )
    dispatch = SimpleNamespace(
        hidden_states=torch.zeros(M, K, dtype=torch.bfloat16),
        topk_output=SimpleNamespace(
            topk_weights=torch.ones(M, TOPK), topk_ids=torch.zeros(M, TOPK, dtype=torch.int64)
        ),
    )
    return calls, obj, layer, dispatch


def test_forwards_moe_semantics_and_scaling(monkeypatch) -> None:
    calls, obj, layer, dispatch = _setup(monkeypatch, scaling=2.5)
    out = fm.fused_moe_txda(obj, layer, dispatch)
    assert calls["activation"] == "silu"
    assert calls["apply_router_weight_on_input"] is False
    assert calls["global_num_experts"] == E
    assert calls["w1_bias"] is None and calls["w2_bias"] is None
    assert torch.equal(out.hidden_states, torch.full((M, N), 2.5))


@pytest.mark.parametrize(
    "kwargs", [{"num_local": E // 2}, {"activation": "gelu"}], ids=["expert_parallel", "gelu"]
)
def test_unsupported_config_raises_before_the_kernel(monkeypatch, kwargs) -> None:
    calls, obj, layer, dispatch = _setup(monkeypatch, **kwargs)
    with pytest.raises(NotImplementedError):
        fm.fused_moe_txda(obj, layer, dispatch)
    assert calls == {}
