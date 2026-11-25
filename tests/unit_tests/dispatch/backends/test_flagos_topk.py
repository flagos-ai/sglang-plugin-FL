import importlib.util
import sys
from collections import namedtuple
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _load_topk_module(monkeypatch, fake_topk_softmax, post_calls, recorder_calls):
    flag_gems_mod = ModuleType("flag_gems")
    flag_gems_mod.topk_softmax = fake_topk_softmax

    def fake_post_process(
        topk_ids,
        topk_weights,
        topk_config,
        router_logits,
        layer_id,
        num_token_non_padded=None,
        expert_location_dispatch_info=None,
    ):
        post_calls.append(
            dict(
                topk_config=topk_config,
                layer_id=layer_id,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
            )
        )
        # SGLang 0.5.18 contract: (ids, weights, recorder_ids).
        return topk_ids, topk_weights, topk_ids.clone()

    sglang_mod = ModuleType("sglang")
    srt_mod = ModuleType("sglang.srt")
    layers_mod = ModuleType("sglang.srt.layers")
    moe_mod = ModuleType("sglang.srt.layers.moe")
    topk_mod = ModuleType("sglang.srt.layers.moe.topk")
    topk_mod.StandardTopKOutput = namedtuple(
        "StandardTopKOutput", "topk_weights topk_ids router_logits"
    )
    topk_mod._post_process_topk_ids = fake_post_process

    eplb_mod = ModuleType("sglang.srt.eplb")
    dispatch_mod = ModuleType("sglang.srt.eplb.expert_location_dispatch")
    dispatch_mod.transform_select_experts_inputs = (
        lambda router_logits, correction_bias, info: (router_logits, correction_bias)
    )
    distribution_mod = ModuleType("sglang.srt.eplb.expert_distribution")
    recorder = SimpleNamespace(
        on_select_experts=lambda topk_ids: recorder_calls.append(topk_ids)
    )
    distribution_mod.get_global_expert_distribution_recorder = lambda: recorder
    eplb_mod.expert_location_dispatch = dispatch_mod
    eplb_mod.expert_distribution = distribution_mod

    for name, mod in {
        "flag_gems": flag_gems_mod,
        "sglang": sglang_mod,
        "sglang.srt": srt_mod,
        "sglang.srt.layers": layers_mod,
        "sglang.srt.layers.moe": moe_mod,
        "sglang.srt.layers.moe.topk": topk_mod,
        "sglang.srt.eplb": eplb_mod,
        "sglang.srt.eplb.expert_location_dispatch": dispatch_mod,
        "sglang.srt.eplb.expert_distribution": distribution_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    module_path = (
        Path(__file__).resolve().parents[4]
        / "sglang_fl"
        / "dispatch"
        / "backends"
        / "flagos"
        / "impl"
        / "topk.py"
    )
    spec = importlib.util.spec_from_file_location("test_flagos_topk_impl", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _topk_config(**overrides):
    cfg = dict(
        top_k=2,
        renormalize=True,
        use_grouped_topk=False,
        custom_routing_function=None,
        correction_bias=None,
        scoring_func="softmax",
        num_fused_shared_experts=0,
        apply_routed_scaling_factor_on_output=False,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def test_flagos_topk_matches_sglang_output_contract(monkeypatch) -> None:
    calls, post_calls, recorder_calls = [], [], []

    def fake_topk_softmax(
        topk_weights, topk_ids, token_expert_indices, router_logits, renormalize
    ):
        calls.append(
            (topk_weights, topk_ids, token_expert_indices, router_logits, renormalize)
        )
        topk_weights.fill_(0.5)
        topk_ids.copy_(torch.tensor([[0, 1], [1, 2]], dtype=torch.int32))
        token_expert_indices.copy_(torch.tensor([[0, 1], [3, 4]], dtype=torch.int32))

    module = _load_topk_module(monkeypatch, fake_topk_softmax, post_calls, recorder_calls)
    router_logits = torch.randn(2, 3)
    hidden_states = torch.randn(2, 4)
    obj = SimpleNamespace(topk_config=_topk_config(), layer_id=7)
    num_non_padded = torch.tensor(2)

    output = module.topk_flagos(
        obj, hidden_states, router_logits, num_token_non_padded=num_non_padded
    )

    assert output.topk_weights.shape == (2, 2)
    assert output.topk_ids.shape == (2, 2)
    assert output.router_logits is router_logits
    assert len(calls) == 1
    weights_arg, ids_arg, scratch_arg, logits_arg, renormalize_arg = calls[0]
    assert weights_arg is output.topk_weights
    assert ids_arg is output.topk_ids
    assert scratch_arg.shape == (2, 2)
    assert scratch_arg.dtype == torch.int32
    assert logits_arg is router_logits
    assert renormalize_arg is True

    # SGLang post-processing runs with the layer context, and the recorder gets
    # the dedicated recorder ids (third return value), not the dispatch ids.
    assert len(post_calls) == 1
    assert post_calls[0]["layer_id"] == 7
    assert post_calls[0]["num_token_non_padded"] is num_non_padded
    assert len(recorder_calls) == 1
    assert recorder_calls[0] is not output.topk_ids
    assert torch.equal(recorder_calls[0], output.topk_ids)


@pytest.mark.parametrize(
    "override",
    [
        {"use_grouped_topk": True},
        {"custom_routing_function": lambda **_: None},
        {"correction_bias": torch.zeros(3)},
        {"scoring_func": "sigmoid"},
        {"num_fused_shared_experts": 1},
        {"apply_routed_scaling_factor_on_output": True},
    ],
)
def test_flagos_topk_rejects_unsupported_routing(monkeypatch, override) -> None:
    def fail_topk_softmax(*args):
        raise AssertionError("kernel must not run for unsupported routing")

    module = _load_topk_module(monkeypatch, fail_topk_softmax, [], [])
    obj = SimpleNamespace(topk_config=_topk_config(**override), layer_id=0)

    # NotImplementedError lets the dispatcher fall back to the next backend.
    with pytest.raises(NotImplementedError):
        module.topk_flagos(obj, torch.randn(2, 4), torch.randn(2, 3))
