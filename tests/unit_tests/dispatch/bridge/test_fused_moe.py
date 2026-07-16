from types import SimpleNamespace

import torch

import sglang_fl.dispatch.bridge.fused_moe as bridge


def test_fused_moe_bridge_forwards_layer_and_dispatch_output(monkeypatch) -> None:
    obj = SimpleNamespace()
    layer = torch.nn.Linear(4, 4)
    dispatch_output = object()
    expected = object()
    calls = []

    def fake_call_op(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(bridge, "call_op", fake_call_op)

    result = bridge.fused_moe_bridge(obj, layer, dispatch_output)

    assert result is expected
    assert calls == [(("fused_moe", obj, layer, dispatch_output), {})]
