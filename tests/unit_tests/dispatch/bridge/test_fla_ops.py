import math

import torch

import sglang_fl.dispatch.bridge.fla_chunk as chunk_bridge
import sglang_fl.dispatch.bridge.fla_fused_recurrent as recurrent_bridge
import sglang_fl.dispatch.bridge.fla_packed_decode as packed_bridge


def test_chunk_gated_delta_rule_bridge_forwards_default_scale(monkeypatch) -> None:
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 5)
    g = torch.randn(1, 2, 3)
    beta = torch.randn(1, 2, 3)
    calls = []

    def fake_call_op(*args, **kwargs):
        calls.append((args, kwargs))
        return "ok"

    monkeypatch.setattr(chunk_bridge, "call_op", fake_call_op)

    result = chunk_bridge.chunk_gated_delta_rule_bridge(q, k, v, g, beta)

    assert result == "ok"
    assert calls[0][0] == ("chunk_gated_delta_rule",)
    kwargs = calls[0][1]
    assert kwargs["q"] is q
    assert kwargs["k"] is k
    assert kwargs["v"] is v
    assert kwargs["g"] is g
    assert kwargs["beta"] is beta
    assert kwargs["scale"] == k.shape[-1] ** -0.5
    assert kwargs["head_first"] is False
    assert kwargs["use_qk_l2norm_in_kernel"] is False


def test_fused_recurrent_bridge_fills_beta_and_contiguous_inputs(monkeypatch) -> None:
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 5)
    g = torch.randn(1, 2, 3).transpose(1, 2)
    calls = []

    def fake_call_op(*args, **kwargs):
        calls.append((args, kwargs))
        return "ok"

    monkeypatch.setattr(recurrent_bridge, "call_op", fake_call_op)

    result = recurrent_bridge.fused_recurrent_gated_delta_rule_bridge(q, k, v, g)

    assert result == "ok"
    assert calls[0][0] == ("fused_recurrent_gated_delta_rule",)
    kwargs = calls[0][1]
    assert kwargs["q"] is q
    assert kwargs["k"] is k
    assert kwargs["v"] is v
    assert kwargs["g"].is_contiguous()
    assert kwargs["beta"].is_contiguous()
    torch.testing.assert_close(kwargs["beta"], torch.ones_like(q[..., 0]))
    assert math.isclose(kwargs["scale"], k.shape[-1] ** -0.5)
    assert kwargs["output_final_state"] is True


def test_fused_recurrent_bridge_keeps_explicit_beta_scale(monkeypatch) -> None:
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 5)
    g = torch.randn(1, 2, 3)
    beta = torch.randn(1, 2, 3)
    calls = []

    monkeypatch.setattr(
        recurrent_bridge,
        "call_op",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "ok",
    )

    recurrent_bridge.fused_recurrent_gated_delta_rule_bridge(
        q,
        k,
        v,
        g,
        beta=beta,
        scale=0.25,
        output_final_state=False,
    )

    kwargs = calls[0][1]
    torch.testing.assert_close(kwargs["beta"], beta)
    assert kwargs["scale"] == 0.25
    assert kwargs["output_final_state"] is False


def test_packed_decode_bridge_forwards_keyword_arguments(monkeypatch) -> None:
    tensors = [torch.randn(2, 3) for _ in range(7)]
    mixed_qkv, a, b, A_log, dt_bias, initial_state, out = tensors
    ssm_state_indices = torch.tensor([0, 1])
    calls = []

    def fake_call_op(*args, **kwargs):
        calls.append((args, kwargs))
        return "ok"

    monkeypatch.setattr(packed_bridge, "call_op", fake_call_op)

    result = packed_bridge.fused_recurrent_gated_delta_rule_packed_decode_bridge(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        0.5,
        initial_state,
        out,
        ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    assert result == "ok"
    assert calls[0][0] == ("fused_recurrent_gated_delta_rule_packed_decode",)
    kwargs = calls[0][1]
    assert kwargs["mixed_qkv"] is mixed_qkv
    assert kwargs["a"] is a
    assert kwargs["b"] is b
    assert kwargs["A_log"] is A_log
    assert kwargs["dt_bias"] is dt_bias
    assert kwargs["scale"] == 0.5
    assert kwargs["initial_state"] is initial_state
    assert kwargs["out"] is out
    assert kwargs["ssm_state_indices"] is ssm_state_indices
    assert kwargs["use_qk_l2norm_in_kernel"] is True
