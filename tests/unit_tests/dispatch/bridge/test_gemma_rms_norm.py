from types import SimpleNamespace

import torch

import sglang_fl.dispatch.bridge.gemma_rms_norm as bridge


def test_gemma_rms_norm_bridge_forwards_to_call_op(monkeypatch) -> None:
    obj = SimpleNamespace()
    x = torch.randn(2, 4)
    residual = torch.randn(2, 4)
    expected = torch.randn(2, 4)
    calls = []

    def fake_call_op(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(bridge, "call_op", fake_call_op)

    result = bridge.gemma_rms_norm_bridge(obj, x, residual=residual)

    assert result is expected
    assert calls == [(("gemma_rms_norm", obj, x, residual), {})]


def test_gemma_rms_norm_bridge_merges_post_residual_addition(monkeypatch) -> None:
    obj = SimpleNamespace()
    x = torch.randn(2, 4)
    residual = torch.randn(2, 4)
    post = torch.randn(2, 4)
    captured = {}

    def fake_call_op(op_name, op_obj, op_x, op_residual):
        captured["args"] = (op_name, op_obj, op_x, op_residual)
        return op_x

    monkeypatch.setattr(bridge, "call_op", fake_call_op)

    bridge.gemma_rms_norm_bridge(
        obj,
        x,
        residual=residual,
        post_residual_addition=post,
    )

    op_name, op_obj, op_x, op_residual = captured["args"]
    assert op_name == "gemma_rms_norm"
    assert op_obj is obj
    assert op_x is x
    torch.testing.assert_close(op_residual, residual + post)
