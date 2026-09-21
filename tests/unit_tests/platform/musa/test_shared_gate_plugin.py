# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace as NS

import pytest
import torch

from sglang_fl.dispatch.backends.vendor.mthreads.patches import (
    shared_expert_gate_tail as patch,
)


def _runtime():
    module = NS(
        get_pp_group=lambda: NS(world_size=1),
        get_moe_data_parallel_world_size=lambda: 1,
        get_attn_context_model_parallel_world_size=lambda: 1,
        get_moe_a2a_backend=lambda: NS(is_none=lambda: True),
    )
    block = NS(tp_size=2, alt_stream=None)
    setattr(block, patch._MODEL_MARKER, True)
    hidden = NS(device=NS(type="musa"), dtype=torch.bfloat16, shape=(4, 2048))
    batch = NS(
        forward_mode=NS(is_decode=lambda: True),
        lora_ids=None,
        mm_inputs=None,
        mm_input_embeds=None,
        spec_info=None,
        spec_algorithm=None,
        tbo_split_seq_index=None,
        tbo_parent_token_range=None,
        tbo_padded_len=None,
        tbo_children=None,
    )
    return module, block, hidden, batch


def test_exact_context_and_empty_optional_lists(monkeypatch):
    monkeypatch.setenv(patch._ENV, "1")
    module, block, hidden, batch = _runtime()
    assert patch._eligible(module, block, hidden, batch, False, False)
    batch.lora_ids, batch.mm_inputs = [None] * 4, (None,) * 4
    assert patch._eligible(module, block, hidden, batch, False, False)


@pytest.mark.parametrize(
    "field,value",
    [
        ("lora_ids", ["adapter"]),
        ("mm_inputs", [object()]),
        ("mm_input_embeds", object()),
        ("spec_info", object()),
        ("spec_algorithm", NS(is_none=lambda: False)),
        ("tbo_split_seq_index", 0),
        ("tbo_parent_token_range", (0, 4)),
        ("tbo_padded_len", 4),
        ("tbo_children", []),
        ("deferred_future_token_ids_map", {}),
        ("deferred_text_mrope_positions", True),
        ("forward_mode", NS(is_decode=lambda: False)),
    ],
)
def test_unsupported_batch_does_not_enter_candidate(monkeypatch, field, value):
    monkeypatch.setenv(patch._ENV, "1")
    module, block, hidden, batch = _runtime()
    setattr(batch, field, value)
    assert not patch._eligible(module, block, hidden, batch, False, False)


@pytest.mark.parametrize(
    "miss",
    [
        "tp",
        "pp",
        "dp",
        "cp",
        "a2a",
        "model",
        "stream",
        "reduce_scatter",
        "fusion",
        "shape",
        "dtype",
        "device",
        "no_batch",
        "disabled",
    ],
)
def test_unsupported_execution_context(monkeypatch, miss):
    monkeypatch.setenv(patch._ENV, "1")
    module, block, hidden, batch = _runtime()
    if miss == "tp":
        block.tp_size = 1
    elif miss == "pp":
        module.get_pp_group = lambda: NS(world_size=2)
    elif miss == "dp":
        module.get_moe_data_parallel_world_size = lambda: 2
    elif miss == "cp":
        module.get_attn_context_model_parallel_world_size = lambda: 2
    elif miss == "a2a":
        module.get_moe_a2a_backend = lambda: NS(is_none=lambda: False)
    elif miss == "model":
        setattr(block, patch._MODEL_MARKER, False)
    elif miss == "stream":
        block.alt_stream = object()
    elif miss == "shape":
        hidden.shape = (8, 2048)
    elif miss == "dtype":
        hidden.dtype = torch.float16
    elif miss == "device":
        hidden.device.type = "cuda"
    elif miss == "no_batch":
        batch = None
    elif miss == "disabled":
        monkeypatch.setenv(patch._ENV, "0")
    assert not patch._eligible(
        module, block, hidden, batch, miss == "reduce_scatter", miss == "fusion"
    )


def _fake_module(events):
    class Block:
        def __init__(self, layer_id, config):
            events.append("init")
            self.shared_expert = object()
            self.shared_expert_gate = object()
            self.fail = False

        def forward(
            self,
            hidden_states,
            forward_batch=None,
            use_reduce_scatter=False,
            should_allreduce_fusion=False,
        ):
            shared = self._forward_shared_experts(hidden_states)
            if self.fail:
                raise RuntimeError("router failed")
            events.extend(["router", "inplace_add", "tp_reduce"])
            return shared

        def _forward_shared_experts(self, hidden_states):
            events.append("original_shared")
            return hidden_states

    return NS(Qwen2MoeSparseMoeBlock=Block)


def test_wrappers_preserve_router_collective_order_and_cleanup(monkeypatch):
    events = []
    module = _fake_module(events)
    cls = module.Qwen2MoeSparseMoeBlock
    monkeypatch.setattr(patch, "_eligible", lambda *args: True)
    sentinel = object()

    def candidate(block, hidden):
        assert patch._ACTIVE_BLOCK.get() is block
        events.append("candidate_shared")
        return sentinel

    monkeypatch.setattr(patch, "_shared_forward", candidate)
    assert patch._patch_block(module)
    initial = (cls.__init__, cls.forward, cls._forward_shared_experts)
    assert patch._patch_block(module)
    assert initial == (cls.__init__, cls.forward, cls._forward_shared_experts)
    obj = cls(0, NS(model_type="qwen3_5_moe_text"))
    assert getattr(obj, patch._MODEL_MARKER)
    assert obj.forward(object()) is sentinel
    assert events == ["init", "candidate_shared", "router", "inplace_add", "tp_reduce"]
    assert patch._ACTIVE_BLOCK.get() is None
    obj.fail = True
    with pytest.raises(RuntimeError, match="router failed"):
        obj.forward(object())
    assert patch._ACTIVE_BLOCK.get() is None
    events.clear()
    hidden = object()
    assert obj._forward_shared_experts(hidden) is hidden
    assert events == ["original_shared"]


def test_wrapper_fallback_and_constructor_marker(monkeypatch):
    events = []
    module = _fake_module(events)
    monkeypatch.setattr(patch, "_eligible", lambda *args: False)
    assert patch._patch_block(module)
    obj = module.Qwen2MoeSparseMoeBlock(layer_id=0, config=NS(model_type="qwen3_5_moe"))
    assert not getattr(obj, patch._MODEL_MARKER)
    hidden = object()
    assert obj.forward(hidden) is hidden
    assert events == ["init", "original_shared", "router", "inplace_add", "tp_reduce"]


def test_existing_shared_override_is_not_replaced():
    module = _fake_module([])
    cls = module.Qwen2MoeSparseMoeBlock
    cls._forward_shared_experts = (
        lambda self, hidden_states, use_musa_gate_tail=False: None
    )
    originals = (cls.__init__, cls.forward, cls._forward_shared_experts)
    assert not patch._patch_block(module)
    assert originals == (cls.__init__, cls.forward, cls._forward_shared_experts)


def test_kernel_guard_miss_keeps_materialization_without_repeating_mlp():
    calls = []
    hidden = torch.zeros((4, 2048), dtype=torch.bfloat16)
    shared = torch.full_like(hidden, 2)
    gate = torch.nn.Linear(2048, 1, bias=False, dtype=torch.bfloat16)
    block = NS(
        shared_expert=lambda x: calls.append(x) or shared, shared_expert_gate=gate
    )
    expected = torch.sigmoid(gate(hidden)) * shared
    actual = patch._shared_forward(block, hidden)
    assert len(calls) == 1
    assert torch.equal(actual, expected)
