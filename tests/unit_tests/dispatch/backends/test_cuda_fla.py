from __future__ import annotations

import torch

from sglang_fl.dispatch.backends.vendor.cuda.impl import fla
from sglang_fl.dispatch import fla_patch


def _tensors():
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 5)
    g = torch.randn(1, 2, 3)
    beta = torch.randn(1, 2, 3)
    return q, k, v, g, beta


def test_chunk_cuda_calls_captured_sglang_function(monkeypatch) -> None:
    q, k, v, g, beta = _tensors()
    initial_state = torch.randn(1, 3, 5, 4)
    initial_state_indices = torch.tensor([0])
    cu_seqlens = torch.tensor([0, 2])
    calls = []

    def native(**kwargs):
        calls.append(kwargs)
        return "native-chunk"

    monkeypatch.setattr(
        fla_patch,
        "get_original",
        lambda name: native if name == "chunk_gated_delta_rule" else None,
    )

    result = fla.chunk_gated_delta_rule_cuda(
        q,
        k,
        v,
        g,
        beta,
        0.5,
        initial_state,
        initial_state_indices,
        cu_seqlens,
        False,
        True,
    )

    assert result == "native-chunk"
    assert calls == [
        {
            "q": q,
            "k": k,
            "v": v,
            "g": g,
            "beta": beta,
            "scale": 0.5,
            "initial_state": initial_state,
            "initial_state_indices": initial_state_indices,
            "cu_seqlens": cu_seqlens,
            "head_first": False,
            "use_qk_l2norm_in_kernel": True,
        }
    ]


def test_fused_recurrent_cuda_uses_v0518_public_contract(monkeypatch) -> None:
    q, k, v, g, beta = _tensors()
    legacy_state_indices = torch.tensor([0])
    legacy_num_accepted = torch.tensor([1])
    calls = []

    def native(**kwargs):
        calls.append(kwargs)
        return "native-recurrent"

    monkeypatch.setattr(
        fla_patch,
        "get_original",
        lambda name: native
        if name == "fused_recurrent_gated_delta_rule"
        else None,
    )

    result = fla.fused_recurrent_gated_delta_rule_cuda(
        q,
        k,
        v,
        g,
        beta,
        0.25,
        None,
        True,
        None,
        legacy_state_indices,
        legacy_num_accepted,
        False,
    )

    assert result == "native-recurrent"
    assert calls == [
        {
            "q": q,
            "k": k,
            "v": v,
            "g": g,
            "beta": beta,
            "scale": 0.25,
            "initial_state": None,
            "output_final_state": True,
            "cu_seqlens": None,
            "use_qk_l2norm_in_kernel": False,
        }
    ]


def test_cuda_fla_requires_patch_to_capture_original(monkeypatch) -> None:
    q, k, v, g, beta = _tensors()
    monkeypatch.setattr(fla_patch, "get_original", lambda name: None)

    try:
        fla.chunk_gated_delta_rule_cuda(q, k, v, g, beta, 0.5)
    except RuntimeError as exc:
        assert "was not captured" in str(exc)
    else:
        raise AssertionError("missing native FLA function should fail clearly")
