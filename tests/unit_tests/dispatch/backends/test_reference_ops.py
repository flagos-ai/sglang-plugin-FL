from types import ModuleType, SimpleNamespace

import torch
import torch.nn.functional as F

from sglang_fl.dispatch.backends.reference.reference import ReferenceBackend


def test_reference_silu_and_mul_matches_pytorch_formula() -> None:
    backend = ReferenceBackend()
    x = torch.randn(3, 8)

    result = backend.silu_and_mul(None, x)
    expected = F.silu(x[..., :4]) * x[..., 4:]

    torch.testing.assert_close(result, expected)


def test_reference_rms_norm_matches_pytorch_formula() -> None:
    backend = ReferenceBackend()
    obj = SimpleNamespace(weight=torch.randn(4), variance_epsilon=1e-6)
    x = torch.randn(3, 4)

    result = backend.rms_norm(obj, x)
    expected = obj.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    torch.testing.assert_close(result, expected)


def test_reference_rms_norm_with_residual_returns_updated_residual() -> None:
    backend = ReferenceBackend()
    obj = SimpleNamespace(weight=torch.randn(4), variance_epsilon=1e-6)
    x = torch.randn(3, 4)
    residual = torch.randn(3, 4)

    output, updated_residual = backend.rms_norm(obj, x, residual)
    expected_residual = x + residual
    expected_output = (
        obj.weight
        * expected_residual
        * torch.rsqrt(expected_residual.pow(2).mean(-1, keepdim=True) + 1e-6)
    )

    torch.testing.assert_close(updated_residual, expected_residual)
    torch.testing.assert_close(output, expected_output)


def test_reference_gemma_rms_norm_uses_weight_plus_one() -> None:
    backend = ReferenceBackend()
    obj = SimpleNamespace(weight=torch.randn(4), variance_epsilon=1e-6)
    x = torch.randn(3, 4)

    result = backend.gemma_rms_norm(obj, x)
    x_float = x.float()
    expected = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + 1e-6)
    expected = (expected * (1.0 + obj.weight.float())).to(x.dtype)

    torch.testing.assert_close(result, expected)


def test_reference_rotary_embedding_matches_neox_formula() -> None:
    backend = ReferenceBackend()
    query = torch.randn(2, 1, 4)
    key = torch.randn(2, 1, 4)
    cos = torch.randn(4, 2)
    sin = torch.randn(4, 2)
    position_ids = torch.tensor([0, 2])

    q_out, k_out = backend.rotary_embedding(
        None,
        query,
        key,
        cos,
        sin,
        position_ids,
        rotary_interleaved=False,
    )

    cos_full = torch.cat([cos[position_ids], cos[position_ids]], dim=-1).unsqueeze(1)
    sin_full = torch.cat([sin[position_ids], sin[position_ids]], dim=-1).unsqueeze(1)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    torch.testing.assert_close(q_out, query * cos_full + rotate_half(query) * sin_full)
    torch.testing.assert_close(k_out, key * cos_full + rotate_half(key) * sin_full)


def test_reference_rotary_embedding_matches_interleaved_formula() -> None:
    backend = ReferenceBackend()
    query = torch.randn(2, 1, 4)
    key = torch.randn(2, 1, 4)
    cos = torch.randn(4, 2)
    sin = torch.randn(4, 2)
    position_ids = torch.tensor([1, 3])

    q_out, k_out = backend.rotary_embedding(
        None,
        query,
        key,
        cos,
        sin,
        position_ids,
        rotary_interleaved=True,
    )

    cos_full = torch.cat([cos[position_ids], cos[position_ids]], dim=-1).unsqueeze(1)
    sin_full = torch.cat([sin[position_ids], sin[position_ids]], dim=-1).unsqueeze(1)

    def rotate_interleaved(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    torch.testing.assert_close(
        q_out,
        query * cos_full + rotate_interleaved(query) * sin_full,
    )
    torch.testing.assert_close(
        k_out,
        key * cos_full + rotate_interleaved(key) * sin_full,
    )


def test_reference_topk_uses_sglang_select_experts_torch_native(monkeypatch) -> None:
    calls = []

    def fake_select_experts(**kwargs):
        calls.append(kwargs)
        return "topk-output"

    sglang_mod = ModuleType("sglang")
    srt_mod = ModuleType("sglang.srt")
    layers_mod = ModuleType("sglang.srt.layers")
    moe_mod = ModuleType("sglang.srt.layers.moe")
    topk_mod = ModuleType("sglang.srt.layers.moe.topk")
    topk_mod.select_experts = fake_select_experts

    monkeypatch.setitem(__import__("sys").modules, "sglang", sglang_mod)
    monkeypatch.setitem(__import__("sys").modules, "sglang.srt", srt_mod)
    monkeypatch.setitem(__import__("sys").modules, "sglang.srt.layers", layers_mod)
    monkeypatch.setitem(__import__("sys").modules, "sglang.srt.layers.moe", moe_mod)
    monkeypatch.setitem(
        __import__("sys").modules,
        "sglang.srt.layers.moe.topk",
        topk_mod,
    )

    backend = ReferenceBackend()
    topk_config = SimpleNamespace(torch_native=False)
    obj = SimpleNamespace(layer_id=7, topk_config=topk_config)
    hidden_states = torch.randn(2, 4)
    router_logits = torch.randn(2, 3)

    result = backend.topk(obj, hidden_states, router_logits)

    assert result == "topk-output"
    assert topk_config.torch_native is True
    assert calls[0]["hidden_states"] is hidden_states
    assert calls[0]["layer_id"] == 7
    assert calls[0]["router_logits"] is router_logits
    assert calls[0]["topk_config"] is topk_config
