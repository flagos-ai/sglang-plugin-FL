# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.utils import musa_patch

PATH = (
    Path(__file__).resolve().parents[3]
    / "sglang_fl/dispatch/backends/vendor"
    / "mthreads/patches/__init__.py"
)
spec = importlib.util.spec_from_file_location("musa_minimax_transaction_test", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_last_installer_failure_restores_all_original_objects():
    owners = [SimpleNamespace(target=object()) for _ in range(3)]
    before = [owner.target for owner in owners]

    def first():
        owners[0].target = object()

    def second():
        owners[1].target = object()

    def fail():
        owners[2].target = object()
        raise RuntimeError("last installer failed")

    with pytest.raises(RuntimeError, match="last installer"):
        module._install_atomically(
            [(owner, "target") for owner in owners], [first, second, fail]
        )
    assert all(owner.target is original for owner, original in zip(owners, before))


def test_missing_required_symbol_prevents_any_installation():
    called = []
    with pytest.raises(AttributeError):
        module._install_atomically(
            [(SimpleNamespace(), "missing")], [lambda: called.append(True)]
        )
    assert called == []


@pytest.mark.parametrize("version", ["0.5.17", "0.5.19", "0.5.18rc1"])
def test_unsupported_release_is_rejected_before_installation(version):
    with pytest.raises(RuntimeError, match="requires SGLang 0.5.18"):
        module.require_supported_sglang(version)


@pytest.mark.parametrize("version", ["0.5.18", "0.5.18+musa"])
def test_supported_release_guard(version):
    module.require_supported_sglang(version)


def test_unsupported_release_keeps_musa_activation_working(monkeypatch, caplog):
    patch = musa_patch.load_musa_patch()
    calls = []
    package = musa_patch.PACKAGE
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.5.19")
    monkeypatch.setitem(sys.modules, f"{package}.patches", module)
    monkeypatch.setitem(
        sys.modules,
        f"{package}.triton_compat",
        SimpleNamespace(patch_triton_pdl_symbols=lambda: calls.append("triton")),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{package}.lifecycle",
        SimpleNamespace(apply_musa_lifecycle_patches=lambda: calls.append("lifecycle")),
    )
    for name in (
        "_patch_gdn_packed_decode_precision",
        "_patch_vision_flash_attention",
        "_patch_pp_send_recv_order",
        "_patch_pp_launch_batch_add_sync",
        "_patch_multimodal_mask",
        "_patch_fp32_tp_all_reduce",
    ):
        monkeypatch.setattr(patch, name, lambda: None)
    # The skip must happen before any version-specific upstream import.
    monkeypatch.setitem(
        sys.modules, "sglang.kernels.ops.attention.minimax_sparse.prefill", None
    )
    patch.apply_musa_patches()
    assert patch._patches_applied
    assert calls == ["triton", "lifecycle"]
    assert "Skipping MUSA MiniMax W8A8 patches" in caplog.text
    assert "0.5.19" in caplog.text
