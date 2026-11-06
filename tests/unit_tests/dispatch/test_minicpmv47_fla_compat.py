# Copyright 2026 FlagOS Contributors

"""Wiring checks for the SGLang FLA interfaces used by MiniCPM-V 4.7."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang_fl.dispatch import fla_patch
from sglang_fl.dispatch.backends.vendor.mthreads.impl import fla as musa_fla
from sglang_fl.dispatch.bridge import fla_chunk


def test_chunk_bridge_forwards_inplace_update(monkeypatch):
    calls = []
    monkeypatch.setattr(fla_chunk, "call_op", lambda *a, **kw: calls.append((a, kw)))
    q = SimpleNamespace(device=SimpleNamespace(type="musa"), shape=(1, 1, 1, 2))
    fla_chunk.chunk_gated_delta_rule_bridge(q, q, q, q, q, inplace_update=False)
    assert calls[0][0] == ("chunk_gated_delta_rule",)
    assert calls[0][1]["inplace_update"] is False


def test_chunk_bridge_keeps_other_backend_signature(monkeypatch):
    calls = []
    monkeypatch.setattr(fla_chunk, "call_op", lambda *a, **kw: calls.append((a, kw)))
    q = torch.empty((1, 1, 1, 2))
    fla_chunk.chunk_gated_delta_rule_bridge(q, q, q, q, q)
    assert "inplace_update" not in calls[0][1]


def test_musa_native_fallback_forwards_inplace_update(monkeypatch):
    calls = []
    monkeypatch.delenv("SGLANG_MUSA_FLAGOS_CHUNK_GDN", raising=False)
    monkeypatch.setattr(musa_fla, "_original", lambda _: lambda **kw: calls.append(kw))
    q = torch.empty((1, 1, 1, 2))
    musa_fla.chunk_gated_delta_rule_musa(q, q, q, q, q, 1.0, inplace_update=False)
    assert calls[0]["inplace_update"] is False


def test_musa_native_fallback_accepts_legacy_signature(monkeypatch):
    calls = []
    monkeypatch.delenv("SGLANG_MUSA_FLAGOS_CHUNK_GDN", raising=False)

    def legacy_original(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state=None,
        initial_state_indices=None,
        cu_seqlens=None,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
    ):
        calls.append((q, k, v, g, beta, scale))

    monkeypatch.setattr(musa_fla, "_original", lambda _: legacy_original)
    q = torch.empty((1, 1, 1, 2))
    musa_fla.chunk_gated_delta_rule_musa(q, q, q, q, q, 1.0)
    assert len(calls) == 1


def test_musa_flagos_state_layout_and_writeback(monkeypatch):
    q = torch.zeros((1, 1, 1, 2))
    state = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2).clone()
    expected_input = state.transpose(-1, -2).clone()
    calls = []

    def fake_flagos(**kw):
        calls.append(kw)
        return q, kw["initial_state"] + 10

    module = ModuleType("flag_gems.fused.chunk_gated_delta_rule")
    module.chunk_gated_delta_rule = fake_flagos
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("SGLANG_MUSA_FLAGOS_CHUNK_GDN", "1")
    result = musa_fla.chunk_gated_delta_rule_musa(
        q,
        q,
        q,
        q,
        q,
        1.0,
        initial_state=state,
        initial_state_indices=torch.tensor([0]),
        cu_seqlens=torch.tensor([0, 1]),
    )
    assert torch.equal(calls[0]["initial_state"], expected_input)
    assert torch.equal(state, (expected_input + 10).transpose(-1, -2))
    assert result[0] is q
    assert result[1:] == (None, None)


def test_fla_patch_is_idempotent(monkeypatch):
    chunk = SimpleNamespace(chunk_gated_delta_rule=object())
    recurrent = SimpleNamespace(
        fused_recurrent_gated_delta_rule=object(),
        fused_recurrent_gated_delta_rule_packed_decode=object(),
    )
    gdn = SimpleNamespace(
        chunk_gated_delta_rule=object(),
        fused_recurrent_gated_delta_rule_packed_decode=object(),
    )
    monkeypatch.setattr(fla_patch, "_patched", False)
    monkeypatch.setattr(fla_patch, "_originals", {})
    monkeypatch.setattr(fla_patch, "_fla_modules", lambda: (chunk, recurrent))
    monkeypatch.setattr(fla_patch.importlib, "import_module", lambda _: gdn)
    original = chunk.chunk_gated_delta_rule
    first = fla_patch.patch_fla_functions()
    assert first["chunk_gated_delta_rule"] is original
    assert fla_patch.patch_fla_functions() is first
    assert chunk.chunk_gated_delta_rule is fla_chunk.chunk_gated_delta_rule_bridge


def test_fla_patch_rolls_back_partial_install(monkeypatch):
    class FailingModule:
        def __init__(self):
            self.fused_recurrent_gated_delta_rule = object()
            self.fused_recurrent_gated_delta_rule_packed_decode = object()

        def __setattr__(self, name, value):
            if name == "fused_recurrent_gated_delta_rule" and callable(value):
                raise RuntimeError("install failed")
            super().__setattr__(name, value)

    chunk = SimpleNamespace(chunk_gated_delta_rule=object())
    recurrent = FailingModule()
    gdn = SimpleNamespace(
        chunk_gated_delta_rule=object(),
        fused_recurrent_gated_delta_rule_packed_decode=object(),
    )
    monkeypatch.setattr(fla_patch, "_patched", False)
    monkeypatch.setattr(fla_patch, "_originals", {})
    monkeypatch.setattr(fla_patch, "_fla_modules", lambda: (chunk, recurrent))
    monkeypatch.setattr(fla_patch.importlib, "import_module", lambda _: gdn)
    original = chunk.chunk_gated_delta_rule
    with pytest.raises(RuntimeError, match="install failed"):
        fla_patch.patch_fla_functions()
    assert chunk.chunk_gated_delta_rule is original
    assert not fla_patch._patched
