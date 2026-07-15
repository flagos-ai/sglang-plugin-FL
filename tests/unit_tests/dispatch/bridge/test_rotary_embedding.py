from types import SimpleNamespace

import torch

import sglang_fl.dispatch.bridge.rotary_embedding as bridge


def test_rotary_embedding_bridge_extracts_cache_and_forwards(monkeypatch) -> None:
    positions = torch.tensor([0, 2])
    query = torch.randn(2, 8)
    key = torch.randn(2, 8)
    cos = torch.randn(4, 4)
    sin = torch.randn(4, 4)
    obj = SimpleNamespace(
        cos_sin_cache=torch.cat([cos, sin], dim=-1),
        head_size=4,
        is_neox_style=True,
    )
    q_embed = torch.randn(2, 2, 4)
    k_embed = torch.randn(2, 2, 4)
    calls = []

    def fake_call_op(*args, **kwargs):
        calls.append((args, kwargs))
        return q_embed, k_embed

    monkeypatch.setattr(bridge, "call_op", fake_call_op)

    out_q, out_k = bridge.rotary_embedding_bridge(obj, positions, query, key)

    args, kwargs = calls[0]
    assert args[0] == "rotary_embedding"
    assert args[1] is obj
    assert args[2].shape == (2, 2, 4)
    assert args[3].shape == (2, 2, 4)
    torch.testing.assert_close(args[4], cos)
    torch.testing.assert_close(args[5], sin)
    assert args[6] is positions
    assert kwargs == {"rotary_interleaved": False, "inplace": True}
    torch.testing.assert_close(out_q, q_embed.reshape_as(query))
    torch.testing.assert_close(out_k, k_embed.reshape_as(key))


def test_rotary_embedding_bridge_applies_offsets(monkeypatch) -> None:
    positions = torch.tensor([0, 2])
    offsets = torch.tensor([1, 1])
    query = torch.randn(2, 8)
    key = torch.randn(2, 8)
    obj = SimpleNamespace(
        cos_sin_cache=torch.randn(4, 8),
        head_size=4,
        is_neox_style=False,
    )
    captured_positions = None

    def fake_call_op(*args, **kwargs):
        nonlocal captured_positions
        captured_positions = args[6]
        return args[2], args[3]

    monkeypatch.setattr(bridge, "call_op", fake_call_op)

    bridge.rotary_embedding_bridge(obj, positions, query, key, offsets=offsets)

    torch.testing.assert_close(captured_positions, positions + offsets)


def test_rotary_embedding_bridge_handles_partial_rotary_dim(monkeypatch) -> None:
    positions = torch.tensor([0, 1])
    query = torch.randn(2, 8)
    key = torch.randn(2, 8)
    pass_q = query.view(2, 2, 4)[..., 2:].clone()
    pass_k = key.view(2, 2, 4)[..., 2:].clone()
    obj = SimpleNamespace(
        cos_sin_cache=torch.randn(4, 2),
        head_size=4,
        is_neox_style=True,
    )

    def fake_call_op(*args, **kwargs):
        q_rot = args[2]
        k_rot = args[3]
        return torch.zeros_like(q_rot), torch.ones_like(k_rot)

    monkeypatch.setattr(bridge, "call_op", fake_call_op)

    out_q, out_k = bridge.rotary_embedding_bridge(obj, positions, query, key)
    out_q = out_q.view(2, 2, 4)
    out_k = out_k.view(2, 2, 4)

    torch.testing.assert_close(out_q[..., :2], torch.zeros_like(out_q[..., :2]))
    torch.testing.assert_close(out_q[..., 2:], pass_q)
    torch.testing.assert_close(out_k[..., :2], torch.ones_like(out_k[..., :2]))
    torch.testing.assert_close(out_k[..., 2:], pass_k)


def test_rotary_embedding_bridge_falls_through_for_fused_kv_arg() -> None:
    expected = (torch.randn(2, 8), torch.randn(2, 8))
    obj = SimpleNamespace(
        forward_native=lambda *args: expected,
        cos_sin_cache=torch.randn(4, 8),
        head_size=4,
    )

    result = bridge.rotary_embedding_bridge(
        obj,
        torch.tensor([0, 1]),
        torch.randn(2, 8),
        torch.randn(2, 8),
        fused_set_kv_buffer_arg=object(),
    )

    assert result is expected
