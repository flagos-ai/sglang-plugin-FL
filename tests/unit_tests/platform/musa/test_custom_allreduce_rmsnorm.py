# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace

from sglang_fl.dispatch.backends.vendor.mthreads.patches import (
    custom_allreduce_rmsnorm as patch,
)


class _Norm:
    variance_epsilon = 1e-6

    def __init__(self):
        self.calls = []

    def forward(self, x, residual, post_residual_addition):
        self.calls.append((x, residual, post_residual_addition))
        return "norm", residual


class _Group:
    world_size = 2

    def __init__(self, fused_result):
        self.fused_result = fused_result
        self.fused_calls = []
        self.all_reduce_calls = []

    def fused_allreduce_rmsnorm(self, x, residual, weight, eps):
        self.fused_calls.append((x, residual, weight, eps))
        return self.fused_result

    def all_reduce(self, x):
        self.all_reduce_calls.append(x)
        return f"reduced:{x}"


def test_fused_result_bypasses_generic_path(monkeypatch):
    group = _Group(("fused-norm", "fused-residual"))
    monkeypatch.setattr(patch, "_select_group", lambda _: group)
    norm = _Norm()

    result = patch._forward_with_musa_allreduce_fusion(
        norm, "x", "residual", None, "weight"
    )

    assert result == ("fused-norm", "fused-residual")
    assert group.all_reduce_calls == []
    assert norm.calls == []


def test_unsupported_shape_keeps_allreduce_then_norm_fallback(monkeypatch):
    group = _Group(None)
    monkeypatch.setattr(patch, "_select_group", lambda _: group)
    norm = _Norm()

    result = patch._forward_with_musa_allreduce_fusion(
        norm, "x", "residual", "post", "weight"
    )

    assert result == ("norm", "residualpost")
    assert group.all_reduce_calls == ["x"]
    assert norm.calls == [("reduced:x", "residualpost", None)]


def test_world_size_one_keeps_original_norm_semantics(monkeypatch):
    group = SimpleNamespace(world_size=1)
    monkeypatch.setattr(patch, "_select_group", lambda _: group)
    norm = _Norm()

    result = patch._forward_with_musa_allreduce_fusion(
        norm, "x", "residual", "post", "weight"
    )

    assert result == ("norm", "residual")
    assert norm.calls == [("x", "residual", "post")]


def test_auto_mode_is_s5000_only(monkeypatch):
    monkeypatch.delenv(patch._ENV_NAME, raising=False)
    monkeypatch.setattr(patch, "_device_is_s5000", lambda: True)
    assert patch._enabled()
    monkeypatch.setattr(patch, "_device_is_s5000", lambda: False)
    assert not patch._enabled()


def test_gate_preserves_original_for_out_of_range_rows():
    original = lambda rows: rows == 0
    assert patch._musa_fusion_gate(original, 64)
    assert patch._musa_fusion_gate(original, 0)
    assert not patch._musa_fusion_gate(original, 131073)
