from types import SimpleNamespace

import torch

import sglang_fl.dispatch.bridge.silu_and_mul as bridge


def test_silu_and_mul_bridge_forwards_to_call_op(monkeypatch) -> None:
    obj = SimpleNamespace()
    x = torch.randn(2, 8)
    expected = torch.randn(2, 4)
    calls = []

    def fake_call_op(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(bridge, "call_op", fake_call_op)

    result = bridge.silu_and_mul_bridge(obj, x)

    assert result is expected
    assert calls == [(("silu_and_mul", obj, x), {})]
