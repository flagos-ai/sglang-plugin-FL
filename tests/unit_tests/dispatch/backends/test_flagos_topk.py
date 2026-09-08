import importlib.util
import sys
from collections import namedtuple
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch


def test_flagos_topk_matches_sglang_output_contract(monkeypatch) -> None:
    calls = []

    def fake_topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indices,
        router_logits,
        renormalize,
    ):
        calls.append(
            (
                topk_weights,
                topk_ids,
                token_expert_indices,
                router_logits,
                renormalize,
            )
        )
        topk_weights.fill_(0.5)
        topk_ids.copy_(torch.tensor([[0, 1], [1, 2]], dtype=torch.int32))
        token_expert_indices.copy_(
            torch.tensor([[0, 1], [3, 4]], dtype=torch.int32)
        )

    flag_gems_mod = ModuleType("flag_gems")
    flag_gems_mod.topk_softmax = fake_topk_softmax

    sglang_mod = ModuleType("sglang")
    srt_mod = ModuleType("sglang.srt")
    layers_mod = ModuleType("sglang.srt.layers")
    moe_mod = ModuleType("sglang.srt.layers.moe")
    topk_mod = ModuleType("sglang.srt.layers.moe.topk")
    topk_mod.StandardTopKOutput = namedtuple(
        "StandardTopKOutput", "topk_weights topk_ids router_logits"
    )

    monkeypatch.setitem(sys.modules, "flag_gems", flag_gems_mod)
    monkeypatch.setitem(sys.modules, "sglang", sglang_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.layers", layers_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.moe", moe_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.moe.topk", topk_mod)
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
    router_logits = torch.randn(2, 3)
    hidden_states = torch.randn(2, 4)
    obj = SimpleNamespace(
        topk_config=SimpleNamespace(top_k=2, renormalize=True),
    )

    output = module.topk_flagos(obj, hidden_states, router_logits)

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
