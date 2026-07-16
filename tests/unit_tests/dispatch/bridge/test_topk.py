from types import SimpleNamespace

import torch

import sglang_fl.dispatch.bridge.topk as bridge


def test_topk_bridge_forwards_kwargs_to_call_op(monkeypatch) -> None:
    obj = SimpleNamespace()
    hidden_states = torch.randn(3, 8)
    router_logits = torch.randn(3, 4)
    num_token_non_padded = torch.tensor([3])
    expert_info = object()
    expected = object()
    calls = []

    def fake_call_op(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(bridge, "call_op", fake_call_op)

    result = bridge.topk_bridge(
        obj,
        hidden_states,
        router_logits,
        num_token_non_padded=num_token_non_padded,
        expert_location_dispatch_info=expert_info,
    )

    assert result is expected
    args, kwargs = calls[0]
    assert args == ("topk", obj, hidden_states, router_logits)
    assert kwargs == {
        "num_token_non_padded": num_token_non_padded,
        "expert_location_dispatch_info": expert_info,
    }
