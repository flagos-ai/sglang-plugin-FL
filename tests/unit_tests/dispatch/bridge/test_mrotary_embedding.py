from types import SimpleNamespace

import torch

import sglang_fl.dispatch.bridge.mrotary_embedding as bridge


def test_mrotary_embedding_bridge_forwards_to_call_op(monkeypatch) -> None:
    obj = SimpleNamespace()
    positions = torch.tensor([[0, 1], [1, 2], [2, 3]])
    query = torch.randn(2, 8)
    key = torch.randn(2, 8)
    expected = (torch.randn(2, 8), torch.randn(2, 8))
    calls = []

    def fake_call_op(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(bridge, "call_op", fake_call_op)

    result = bridge.mrotary_embedding_bridge(obj, positions, query, key)

    assert result is expected
    assert calls == [(("mrotary_embedding", obj, positions, query, key), {})]


def test_mrotary_embedding_bridge_falls_through_for_fused_kv_arg() -> None:
    expected = (torch.randn(2, 8), torch.randn(2, 8))
    obj = SimpleNamespace(forward_native=lambda *args: expected)

    result = bridge.mrotary_embedding_bridge(
        obj,
        torch.tensor([0, 1]),
        torch.randn(2, 8),
        torch.randn(2, 8),
        fused_set_kv_buffer_arg=object(),
    )

    assert result is expected
