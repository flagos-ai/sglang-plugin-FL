# Copyright (c) 2026 BAAI. All rights reserved.

"""Unit tests for the MTT S5000 FLA adapters."""

from __future__ import annotations

from unittest.mock import Mock

import torch


def _packed_inputs(*, state_dtype=torch.float32):
    batch, qk_heads, value_heads, dim = 2, 2, 4, 128
    packed_dim = 2 * qk_heads * dim + value_heads * dim
    return {
        "mixed_qkv": torch.randn(batch, packed_dim, dtype=torch.bfloat16),
        "a": torch.randn(batch, value_heads, dtype=torch.bfloat16),
        "b": torch.randn(batch, value_heads, dtype=torch.bfloat16),
        "A_log": torch.randn(value_heads, dtype=torch.float32),
        "dt_bias": torch.randn(value_heads, dtype=torch.float32),
        "scale": dim**-0.5,
        "initial_state": torch.randn(8, value_heads, dim, dim, dtype=state_dtype),
        "out": torch.empty(batch, 1, value_heads, dim, dtype=torch.bfloat16),
        "ssm_state_indices": torch.tensor([1, 5], dtype=torch.int64),
        "use_qk_l2norm_in_kernel": True,
    }


def _prefill_inputs(*, gate_dtype=torch.float32):
    total, qk_heads, value_heads, dim = 65, 2, 4, 128
    return {
        "q": torch.randn(1, total, qk_heads, dim, dtype=torch.bfloat16),
        "k": torch.randn(1, total, qk_heads, dim, dtype=torch.bfloat16),
        "v": torch.randn(1, total, value_heads, dim, dtype=torch.bfloat16),
        "g": torch.randn(1, total, value_heads, dtype=gate_dtype),
        "beta": torch.rand(1, total, value_heads, dtype=gate_dtype),
        "scale": dim**-0.5,
        "initial_state": torch.randn(8, value_heads, dim, dim, dtype=torch.float32),
        "initial_state_indices": torch.tensor([1, 5], dtype=torch.int64),
        "cu_seqlens": torch.tensor([0, 32, total], dtype=torch.int32),
        "head_first": False,
        "use_qk_l2norm_in_kernel": True,
    }


def test_mate_chunk_prefill_gathers_and_writes_back_state(monkeypatch) -> None:
    from sglang_fl.dispatch.backends.vendor.mthreads.impl import fla

    inputs = _prefill_inputs()
    original_state = inputs["initial_state"].clone()
    captured = {}

    def fake_mate_gdn_prefill(**kwargs):
        captured.update(kwargs)
        output = torch.full_like(kwargs["v"], 3)
        final_state = torch.full_like(kwargs["initial_state"], 7)
        return output, final_state

    monkeypatch.setattr(fla, "_is_s5000", lambda device: True)
    monkeypatch.setattr(fla, "_mamba_radix_cache_disabled", lambda: True)
    monkeypatch.setattr(fla, "_load_mate_gdn_prefill", lambda: fake_mate_gdn_prefill)

    output, final_state, h = fla.chunk_gated_delta_rule_musa(**inputs)

    assert output.shape == inputs["v"].shape
    assert torch.all(output == 3)
    assert final_state is None
    assert h is None
    assert captured["q"].shape == (65, 2, 128)
    assert captured["k"].shape == (65, 2, 128)
    assert captured["v"].shape == (65, 4, 128)
    assert captured["g"].shape == (65, 4)
    assert captured["beta"].shape == (65, 4)
    assert captured["initial_state"].shape == (2, 4, 128, 128)
    assert captured["cu_seqlens"] is inputs["cu_seqlens"]
    assert captured["use_qk_l2norm_in_kernel"] is True
    assert captured["chunk_size"] == 64
    assert captured["is_log_space"] is True
    assert torch.all(inputs["initial_state"][[1, 5]] == 7)
    assert torch.equal(inputs["initial_state"][[0, 2]], original_state[[0, 2]])


def test_mate_chunk_prefill_falls_back_when_radix_cache_is_enabled(
    monkeypatch,
) -> None:
    from sglang_fl.dispatch.backends.vendor.mthreads.impl import fla

    inputs = _prefill_inputs()
    fallback_result = (torch.empty(1), torch.empty(1), torch.empty(1))
    original = Mock(return_value=fallback_result)

    monkeypatch.setattr(fla, "_is_s5000", lambda device: True)
    monkeypatch.setattr(fla, "_mamba_radix_cache_disabled", lambda: False)
    monkeypatch.setattr(fla, "_load_mate_gdn_prefill", Mock())
    monkeypatch.setattr(fla, "_original", lambda name: original)

    assert fla.chunk_gated_delta_rule_musa(**inputs) is fallback_result
    fla._load_mate_gdn_prefill.assert_not_called()
    original.assert_called_once_with(**inputs)


def test_mate_chunk_prefill_falls_back_for_non_fp32_gates(monkeypatch) -> None:
    from sglang_fl.dispatch.backends.vendor.mthreads.impl import fla

    inputs = _prefill_inputs(gate_dtype=torch.bfloat16)
    fallback_result = (torch.empty(1), torch.empty(1), torch.empty(1))
    original = Mock(return_value=fallback_result)

    monkeypatch.setattr(fla, "_is_s5000", lambda device: True)
    monkeypatch.setattr(fla, "_mamba_radix_cache_disabled", lambda: True)
    monkeypatch.setattr(fla, "_load_mate_gdn_prefill", Mock())
    monkeypatch.setattr(fla, "_original", lambda name: original)

    assert fla.chunk_gated_delta_rule_musa(**inputs) is fallback_result
    fla._load_mate_gdn_prefill.assert_not_called()
    original.assert_called_once_with(**inputs)


def test_mate_packed_decode_uses_strided_views_and_caller_buffers(monkeypatch) -> None:
    from sglang_fl.dispatch.backends.vendor.mthreads.impl import fla

    inputs = _packed_inputs()
    captured = {}

    def fake_mate_gdn_decode(**kwargs):
        captured.update(kwargs)
        kwargs["output"].fill_(3)
        return kwargs["output"], kwargs["state"]

    monkeypatch.setattr(fla, "_is_s5000", lambda device: True)
    monkeypatch.setattr(fla, "_load_mate_gdn_decode", lambda: fake_mate_gdn_decode)

    result, state = fla.fused_recurrent_gated_delta_rule_packed_decode_musa(**inputs)

    assert result is inputs["out"]
    assert state is inputs["initial_state"]
    assert captured["q"].shape == (2, 1, 2, 128)
    assert captured["k"].shape == (2, 1, 2, 128)
    assert captured["v"].shape == (2, 1, 4, 128)
    assert captured["a"].shape == (2, 1, 4)
    assert captured["b"].shape == (2, 1, 4)
    assert captured["q"]._base is not None
    assert captured["k"]._base is not None
    assert captured["v"]._base is not None
    assert captured["state_layout"] == "VK"
    assert captured["state_indices"] is inputs["ssm_state_indices"]
    assert captured["scale"] == inputs["scale"]
    assert captured["output"] is inputs["out"]
    assert captured["use_qk_l2norm"] is True
    assert torch.all(result == 3)


def test_mate_packed_decode_falls_back_for_unsupported_state(monkeypatch) -> None:
    from sglang_fl.dispatch.backends.vendor.mthreads.impl import fla

    inputs = _packed_inputs(state_dtype=torch.bfloat16)
    fallback_result = (torch.empty(1), torch.empty(1))
    original = Mock(return_value=fallback_result)

    monkeypatch.setattr(fla, "_is_s5000", lambda device: True)
    monkeypatch.setattr(fla, "_load_mate_gdn_decode", Mock())
    monkeypatch.setattr(fla, "_original", lambda name: original)

    assert (
        fla.fused_recurrent_gated_delta_rule_packed_decode_musa(**inputs)
        is fallback_result
    )
    fla._load_mate_gdn_decode.assert_not_called()
    original.assert_called_once_with(**inputs)


def test_mate_packed_decode_can_be_disabled(monkeypatch) -> None:
    from sglang_fl.dispatch.backends.vendor.mthreads.impl import fla

    inputs = _packed_inputs()
    fallback_result = (torch.empty(1), torch.empty(1))
    original = Mock(return_value=fallback_result)

    monkeypatch.setenv("SGLANG_MUSA_MATE_GDN", "off")
    monkeypatch.setattr(fla, "_is_s5000", lambda device: True)
    monkeypatch.setattr(fla, "_load_mate_gdn_decode", Mock())
    monkeypatch.setattr(fla, "_original", lambda name: original)

    assert (
        fla.fused_recurrent_gated_delta_rule_packed_decode_musa(**inputs)
        is fallback_result
    )
    fla._load_mate_gdn_decode.assert_not_called()
    original.assert_called_once_with(**inputs)
