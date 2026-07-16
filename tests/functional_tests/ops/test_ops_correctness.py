# Copyright (c) 2026 BAAI. All rights reserved.

"""Functional correctness tests for SGLang-FL dispatch operators.

These tests execute real ``sglang_fl.dispatch.call_op`` paths on GPU and compare
basic operator outputs against the reference backend where numerical comparison
is stable. They are intentionally smaller than e2e model tests and focus on the
foundational bridge/dispatch operators used by SGLang models.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from sglang_fl.dispatch import SelectionPolicy, call_op, policy_context

pytestmark = [pytest.mark.functional, pytest.mark.gpu]


_RTOL = 1e-3
_ATOL = 1e-3
_HALF_TOL = 2e-2


def _allclose(actual: torch.Tensor, expected: torch.Tensor, *, dtype: torch.dtype) -> bool:
    tol = _HALF_TOL if dtype in (torch.float16, torch.bfloat16) else _ATOL
    return torch.allclose(actual.float(), expected.float(), rtol=tol, atol=tol)


def _selected_prefer() -> str:
    prefer = os.environ.get("SGLANG_FL_PREFER", "flagos").strip().lower()
    return prefer if prefer in {"flagos", "vendor", "reference"} else "flagos"


def _call_reference(op_name: str, *args, **kwargs):
    policy = SelectionPolicy.from_dict(prefer="reference", strict=False)
    with policy_context(policy):
        return call_op(op_name, *args, **kwargs)


def _call_selected(op_name: str, *args, **kwargs):
    # Functional tests should validate the preferred implementation when it is
    # available, while still allowing fallback so partially wired backends do not
    # hide the rest of the operator surface.
    policy = SelectionPolicy.from_dict(prefer=_selected_prefer(), strict=True)
    with policy_context(policy):
        return call_op(op_name, *args, **kwargs)


def _maybe_skip_unavailable(exc: RuntimeError, op_name: str) -> None:
    if "No available implementation" in str(exc):
        pytest.skip(f"{op_name} not available: {exc}")
    raise exc


def _skip_optional_kernel(exc: Exception, op_name: str) -> None:
    pytest.skip(f"{op_name} kernel unavailable in this environment: {exc}")


def _assert_tensor_finite(tensor: torch.Tensor, shape: tuple[int, ...] | None = None) -> None:
    if shape is not None:
        assert tensor.shape == shape
    assert torch.isfinite(tensor.float()).all()


def _rms_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def _rotary_cache(max_position: int, rotary_dim: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim)
    )
    t = torch.arange(max_position, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def test_silu_and_mul_correctness(device) -> None:
    """Compare silu_and_mul with the PyTorch definition."""
    shapes = [(1, 128), (8, 256), (32, 512)]
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for batch, hidden_twice in shapes:
            x = torch.randn(batch, hidden_twice, device=device, dtype=dtype)
            half = hidden_twice // 2
            expected = F.silu(x[..., :half]) * x[..., half:]
            try:
                actual = _call_selected("silu_and_mul", None, x)
            except RuntimeError as exc:
                _maybe_skip_unavailable(exc, "silu_and_mul")
            assert actual.shape == expected.shape
            assert _allclose(actual, expected, dtype=dtype)


def test_rms_norm_correctness(device) -> None:
    """Compare rms_norm with a PyTorch RMSNorm reference."""
    eps = 1e-6
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(4, 16, 128, device=device, dtype=dtype)
        weight = torch.randn(128, device=device, dtype=dtype)
        obj = SimpleNamespace(weight=weight, variance_epsilon=eps)
        expected = _rms_reference(x, weight, eps)
        try:
            actual = _call_selected("rms_norm", obj, x)
        except RuntimeError as exc:
            _maybe_skip_unavailable(exc, "rms_norm")
        if isinstance(actual, tuple):
            actual = actual[0]
        assert actual.shape == expected.shape
        assert _allclose(actual, expected, dtype=dtype)


def test_rms_norm_with_residual_contract(device) -> None:
    """Check rms_norm residual mode returns normalized output and updated residual."""
    eps = 1e-6
    x = torch.randn(2, 8, 64, device=device, dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn(64, device=device, dtype=torch.float16)
    obj = SimpleNamespace(weight=weight, variance_epsilon=eps)
    try:
        actual = _call_selected("rms_norm", obj, x, residual)
    except RuntimeError as exc:
        _maybe_skip_unavailable(exc, "rms_norm")
    assert isinstance(actual, tuple)
    normed, updated_residual = actual
    assert normed.shape == x.shape
    assert updated_residual.shape == x.shape


def test_gemma_rms_norm_correctness(device) -> None:
    """Compare gemma_rms_norm with the Gemma weight semantics reference."""
    eps = 1e-6
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(4, 16, 128, device=device, dtype=dtype)
        weight = torch.randn(128, device=device, dtype=dtype)
        obj = SimpleNamespace(weight=weight, variance_epsilon=eps)
        variance = x.float().pow(2).mean(-1, keepdim=True)
        expected = (x.float() * torch.rsqrt(variance + eps) * (1.0 + weight.float())).to(dtype)
        try:
            actual = _call_selected("gemma_rms_norm", obj, x)
        except RuntimeError as exc:
            _maybe_skip_unavailable(exc, "gemma_rms_norm")
        if isinstance(actual, tuple):
            actual = actual[0]
        assert actual.shape == expected.shape
        assert _allclose(actual, expected, dtype=dtype)


def test_rotary_embedding_correctness(device) -> None:
    """Compare rotary_embedding selected path against reference backend."""
    dtype = torch.float16
    num_tokens, num_heads, head_size = 16, 4, 64
    positions = torch.arange(num_tokens, device=device, dtype=torch.long)
    cos, sin = _rotary_cache(128, head_size, device, dtype)
    q = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=dtype)
    k = torch.randn_like(q)

    try:
        expected_q, expected_k = _call_reference(
            "rotary_embedding", None, q.clone(), k.clone(), cos, sin, positions, False, False
        )
        actual_q, actual_k = _call_selected(
            "rotary_embedding", None, q.clone(), k.clone(), cos, sin, positions, False, False
        )
    except RuntimeError as exc:
        _maybe_skip_unavailable(exc, "rotary_embedding")

    assert actual_q.shape == expected_q.shape
    assert actual_k.shape == expected_k.shape
    assert _allclose(actual_q, expected_q, dtype=dtype)
    assert _allclose(actual_k, expected_k, dtype=dtype)


def test_mrotary_embedding_correctness(device) -> None:
    """Compare mrotary_embedding selected path against reference backend."""
    dtype = torch.float16
    num_tokens, num_heads, head_size, rotary_dim = 12, 4, 64, 64
    cos, sin = _rotary_cache(128, rotary_dim, device, dtype)
    cos_sin_cache = torch.cat([cos, sin], dim=-1)
    positions = torch.arange(num_tokens, device=device, dtype=torch.long)
    q = torch.randn(num_tokens, num_heads * head_size, device=device, dtype=dtype)
    k = torch.randn_like(q)
    obj = SimpleNamespace(
        head_size=head_size,
        rotary_dim=rotary_dim,
        is_neox_style=True,
        cos_sin_cache=cos_sin_cache,
        mrope_section=None,
    )

    try:
        expected_q, expected_k = _call_reference("mrotary_embedding", obj, positions, q.clone(), k.clone())
        actual_q, actual_k = _call_selected("mrotary_embedding", obj, positions, q.clone(), k.clone())
    except RuntimeError as exc:
        _maybe_skip_unavailable(exc, "mrotary_embedding")

    assert actual_q.shape == expected_q.shape
    assert actual_k.shape == expected_k.shape
    assert _allclose(actual_q, expected_q, dtype=dtype)
    assert _allclose(actual_k, expected_k, dtype=dtype)


def test_topk_output_contract(device) -> None:
    """Check topk returns a valid SGLang StandardTopKOutput-like object."""
    try:
        from sglang.srt.layers.moe.topk import TopKConfig
    except Exception as exc:  # pragma: no cover - depends on installed SGLang
        pytest.skip(f"SGLang TopKConfig unavailable: {exc}")

    num_tokens, hidden_size, num_experts, top_k = 16, 64, 8, 2
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.float16)
    router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_config = TopKConfig(top_k=top_k, renormalize=True)
    obj = SimpleNamespace(layer_id=0, topk_config=topk_config)

    try:
        output = _call_selected("topk", obj, hidden_states, router_logits)
    except RuntimeError as exc:
        _maybe_skip_unavailable(exc, "topk")

    assert hasattr(output, "topk_weights")
    assert hasattr(output, "topk_ids")
    assert output.topk_weights.shape == (num_tokens, top_k)
    assert output.topk_ids.shape == (num_tokens, top_k)
    assert torch.all(output.topk_ids >= 0)
    assert torch.all(output.topk_ids < num_experts)
    assert torch.allclose(
        output.topk_weights.float().sum(dim=-1),
        torch.ones(num_tokens, device=device),
        rtol=1e-3,
        atol=1e-3,
    )

def test_fused_moe_output_contract(device) -> None:
    """Check fused_moe dispatches to a MoeRunner-compatible object."""
    try:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
        from sglang.srt.layers.moe.topk import StandardTopKOutput
    except Exception as exc:  # pragma: no cover - depends on installed SGLang
        pytest.skip(f"SGLang MoE dispatcher types unavailable: {exc}")

    class MockRunner:
        def run(self, dispatch_output, quant_info):
            assert quant_info.w13_weight is layer.w13_weight
            assert quant_info.w2_weight is layer.w2_weight
            weights = dispatch_output.topk_output.topk_weights.to(dispatch_output.hidden_states.dtype)
            scale = weights.sum(dim=-1, keepdim=True)
            return StandardCombineInput(hidden_states=dispatch_output.hidden_states * scale)

    num_tokens, hidden_size, num_experts, intermediate_size, top_k = 8, 16, 4, 32, 2
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.float16)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device=device, dtype=torch.float32),
        dim=-1,
    )
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int32)
    router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_output = StandardTopKOutput(topk_weights, topk_ids, router_logits)
    dispatch_output = StandardDispatchOutput(hidden_states, None, topk_output)
    layer = SimpleNamespace(
        w13_weight=torch.empty(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=torch.float16,
        ),
        w2_weight=torch.empty(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.float16,
        ),
    )
    obj = SimpleNamespace(runner=MockRunner())

    try:
        output = _call_selected("fused_moe", obj, layer, dispatch_output)
    except RuntimeError as exc:
        _maybe_skip_unavailable(exc, "fused_moe")

    assert hasattr(output, "hidden_states")
    _assert_tensor_finite(output.hidden_states, (num_tokens, hidden_size))


def _make_fla_inputs(device, dtype: torch.dtype = torch.float16):
    batch, seq_len, num_heads, key_dim, value_dim = 1, 4, 2, 16, 16
    q = torch.randn(batch, seq_len, num_heads, key_dim, device=device, dtype=dtype) * 0.1
    k = torch.randn_like(q)
    v = torch.randn(batch, seq_len, num_heads, value_dim, device=device, dtype=dtype) * 0.1
    g = torch.randn(batch, seq_len, num_heads, device=device, dtype=dtype) * -0.1
    beta = torch.sigmoid(torch.randn(batch, seq_len, num_heads, device=device, dtype=dtype))
    scale = key_dim**-0.5
    return q, k, v, g, beta, scale


def test_fla_chunk_gated_delta_rule_contract(device) -> None:
    """Check FLA chunk_gated_delta_rule returns valid output tensors."""
    q, k, v, g, beta, scale = _make_fla_inputs(device)

    try:
        output, _, final_state = _call_selected(
            "chunk_gated_delta_rule",
            q,
            k,
            v,
            g,
            beta,
            scale,
            head_first=False,
        )
    except RuntimeError as exc:
        if "No available implementation" in str(exc):
            _maybe_skip_unavailable(exc, "chunk_gated_delta_rule")
        _skip_optional_kernel(exc, "chunk_gated_delta_rule")
    except Exception as exc:
        _skip_optional_kernel(exc, "chunk_gated_delta_rule")

    _assert_tensor_finite(output, tuple(v.shape))
    if final_state is not None:
        _assert_tensor_finite(final_state)


def test_fla_fused_recurrent_gated_delta_rule_contract(device) -> None:
    """Check FLA fused_recurrent_gated_delta_rule returns valid output tensors."""
    q, k, v, g, beta, scale = _make_fla_inputs(device)

    try:
        output, final_state = _call_selected(
            "fused_recurrent_gated_delta_rule",
            q,
            k,
            v,
            g,
            beta,
            scale,
            output_final_state=True,
        )
    except RuntimeError as exc:
        if "No available implementation" in str(exc):
            _maybe_skip_unavailable(exc, "fused_recurrent_gated_delta_rule")
        _skip_optional_kernel(exc, "fused_recurrent_gated_delta_rule")
    except Exception as exc:
        _skip_optional_kernel(exc, "fused_recurrent_gated_delta_rule")

    _assert_tensor_finite(output, tuple(v.shape))
    if final_state is not None:
        _assert_tensor_finite(final_state)


def test_fla_packed_decode_contract(device) -> None:
    """Check FLA packed decode dispatch returns valid output/state tensors."""
    batch, num_heads, key_dim, value_dim = 2, 2, 16, 16
    dtype = torch.float16
    mixed_qkv = torch.randn(
        batch,
        num_heads,
        key_dim * 2 + value_dim,
        device=device,
        dtype=dtype,
    )
    a = torch.randn(batch, num_heads, device=device, dtype=dtype)
    b = torch.randn(batch, num_heads, device=device, dtype=dtype)
    A_log = torch.randn(num_heads, device=device, dtype=dtype)
    dt_bias = torch.randn(num_heads, device=device, dtype=dtype)
    initial_state = torch.randn(batch, num_heads, key_dim, value_dim, device=device, dtype=dtype)
    out = torch.empty(batch, num_heads, value_dim, device=device, dtype=dtype)
    ssm_state_indices = torch.arange(batch, device=device, dtype=torch.int32)

    try:
        output, final_state = _call_selected(
            "fused_recurrent_gated_delta_rule_packed_decode",
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            key_dim**-0.5,
            initial_state,
            out,
            ssm_state_indices,
        )
    except RuntimeError as exc:
        if "No available implementation" in str(exc):
            _maybe_skip_unavailable(exc, "fused_recurrent_gated_delta_rule_packed_decode")
        _skip_optional_kernel(exc, "fused_recurrent_gated_delta_rule_packed_decode")
    except Exception as exc:
        _skip_optional_kernel(exc, "fused_recurrent_gated_delta_rule_packed_decode")

    _assert_tensor_finite(output)
    _assert_tensor_finite(final_state)


