# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

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


@pytest.mark.parametrize("version", ["0.5.17", "0.5.19", "0.5.18+vendor"])
def test_unsupported_release_is_rejected_before_installation(version):
    with pytest.raises(RuntimeError, match="requires SGLang 0.5.18"):
        module.require_supported_sglang(version)


def test_supported_release_guard():
    module.require_supported_sglang("0.5.18")
