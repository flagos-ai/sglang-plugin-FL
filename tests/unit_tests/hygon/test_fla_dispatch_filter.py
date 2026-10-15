# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for restoring OOT-filtered Hygon FLA bindings."""

from types import SimpleNamespace

import pytest

from sglang_fl.dispatch.backends.vendor.hygon.patches import (
    fla_dispatch_filter as patch,
)


@pytest.fixture(autouse=True)
def filters(monkeypatch):
    monkeypatch.delenv("SGLANG_FL_OOT_WHITELIST", raising=False)
    monkeypatch.delenv("SGLANG_FL_OOT_BLACKLIST", raising=False)
    monkeypatch.setattr(patch, "get_effective_config", lambda: {})


@pytest.mark.parametrize(
    "namespace",
    [
        "sglang.kernels.ops.attention.fla",
        "sglang.srt.layers.attention.fla",
    ],
)
def test_restores_filtered_public_and_gdn_functions(monkeypatch, namespace):
    chunk, recurrent, packed = object(), object(), object()
    originals = {
        "chunk_gated_delta_rule": chunk,
        "fused_recurrent_gated_delta_rule": recurrent,
        "fused_recurrent_gated_delta_rule_packed_decode": packed,
    }
    modules = {
        f"{namespace}.chunk": SimpleNamespace(chunk_gated_delta_rule=object()),
        f"{namespace}.fused_recurrent": SimpleNamespace(
            fused_recurrent_gated_delta_rule=object(),
            fused_recurrent_gated_delta_rule_packed_decode=object(),
        ),
        "sglang.srt.layers.attention.linear.kernels.gdn_triton": SimpleNamespace(
            chunk_gated_delta_rule=object(),
            fused_recurrent_gated_delta_rule_packed_decode=object(),
        ),
    }
    recurrent_module = modules[f"{namespace}.fused_recurrent"]
    unfiltered = recurrent_module.fused_recurrent_gated_delta_rule

    def import_module(name):
        if name not in modules:
            raise ModuleNotFoundError(name, name=name)
        return modules[name]

    monkeypatch.setattr(patch.importlib, "import_module", import_module)
    monkeypatch.setattr(patch, "get_original", originals.get)
    monkeypatch.setenv(
        "SGLANG_FL_OOT_BLACKLIST",
        "chunk_gated_delta_rule,fused_recurrent_gated_delta_rule_packed_decode",
    )
    expected = (
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule_packed_decode",
    )
    assert patch.patch_fla_dispatch_filter() == expected
    assert patch.patch_fla_dispatch_filter() == expected
    assert modules[f"{namespace}.chunk"].chunk_gated_delta_rule is chunk
    assert recurrent_module.fused_recurrent_gated_delta_rule_packed_decode is packed
    assert recurrent_module.fused_recurrent_gated_delta_rule is unfiltered
    gdn = modules["sglang.srt.layers.attention.linear.kernels.gdn_triton"]
    assert gdn.chunk_gated_delta_rule is chunk
    assert gdn.fused_recurrent_gated_delta_rule_packed_decode is packed


def test_whitelist_restores_other_fla_functions(monkeypatch):
    monkeypatch.setenv("SGLANG_FL_OOT_WHITELIST", "rms_norm,chunk_gated_delta_rule")
    whitelist, blacklist = patch._get_oot_filters()
    assert not patch._is_filtered("chunk_gated_delta_rule", whitelist, blacklist)
    assert patch._is_filtered("fused_recurrent_gated_delta_rule", whitelist, blacklist)


def test_inactive_common_patch_does_not_import_fla(monkeypatch):
    monkeypatch.setenv("SGLANG_FL_OOT_BLACKLIST", "chunk_gated_delta_rule")
    monkeypatch.setattr(patch, "get_original", lambda name: None)

    def unexpected_import(name):
        pytest.fail("Inactive common FLA patch must not import GPU kernels")

    monkeypatch.setattr(patch.importlib, "import_module", unexpected_import)
    assert patch.patch_fla_dispatch_filter() == ()


def test_missing_kernel_dependency_does_not_fall_back_to_legacy_namespace(monkeypatch):
    calls = []

    def import_module(name):
        calls.append(name)
        raise ModuleNotFoundError("No module named 'triton'", name="triton")

    monkeypatch.setattr(patch.importlib, "import_module", import_module)
    with pytest.raises(ModuleNotFoundError, match="triton"):
        patch._import_fla_module("chunk")
    assert calls == ["sglang.kernels.ops.attention.fla.chunk"]
