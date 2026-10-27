"""Tests for the PPU (thead) import placeholders (flashinfer, deep_gemm)."""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import types

import pytest

from sglang_fl.dispatch.backends.vendor.thead import import_guard as guard

ROOTS = list(guard._PACKAGES)
_REAL_INSTALLED = guard._installed  # the fixture stubs it out; one test needs the real one


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.setattr(
        sys,
        "meta_path",
        [f for f in sys.meta_path if not isinstance(f, guard._PlaceholderFinder)],
    )
    for root in ROOTS:
        for name in [m for m in sys.modules if m == root or m.startswith(root + ".")]:
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.delenv(guard._PACKAGES[root][0], raising=False)
        for key in guard._PACKAGES[root][1]:
            monkeypatch.setenv(key, "true")
    monkeypatch.setattr(guard, "_installed", lambda root: False)
    yield
    for root in ROOTS:
        for name in [m for m in sys.modules if m == root or m.startswith(root + ".")]:
            sys.modules.pop(name, None)


def _active(root):
    return guard._active_finder(root) is not None


def test_guards_both_packages_and_forces_availability_off(clean):
    assert guard.install_import_guards() == ["flashinfer", "deep_gemm"]
    assert os.environ["SGLANG_IS_FLASHINFER_AVAILABLE"] == "false"
    assert os.environ["SGLANG_ENABLE_JIT_DEEPGEMM"] == "false"


def test_module_level_import_patterns(clean):
    guard.install_import_guards()

    from flashinfer import bmm_fp8  # noqa: F401  (fp8_utils.py pattern)
    import deep_gemm  # noqa: F401  (dsa_backend.py pattern)
    import deep_gemm.utils.layout  # noqa: F401  (fp8.py pattern)
    from deep_gemm import get_paged_mqa_logits_metadata
    import flashinfer.fused_moe.core as core

    class Derived(core.SomeBase):  # subclassing at import time must work
        pass

    assert getattr(sys.modules["deep_gemm"], "__sglang_fl_placeholder__", False)
    with pytest.raises(RuntimeError, match="deep_gemm is not installed"):
        get_paged_mqa_logits_metadata()


def test_install_is_idempotent(clean):
    guard.install_import_guards()
    guard.install_import_guards()
    for root in ROOTS:
        count = sum(
            isinstance(f, guard._PlaceholderFinder) and f.root == root
            for f in sys.meta_path
        )
        assert count == 1


def test_installed_package_is_never_shadowed(clean, monkeypatch):
    real = types.ModuleType("deep_gemm")
    real.__path__ = []

    class _RealFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path, target=None):
            if fullname == "deep_gemm":
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
            return None

        def create_module(self, spec):
            return real

        def exec_module(self, module):
            pass

    monkeypatch.setattr(sys, "meta_path", [_RealFinder()] + sys.meta_path)
    monkeypatch.setattr(guard, "_installed", _REAL_INSTALLED)

    assert guard.install_placeholder("deep_gemm") is False
    assert not _active("deep_gemm")
    assert importlib.import_module("deep_gemm") is real
    assert os.environ["SGLANG_ENABLE_JIT_DEEPGEMM"] == "true"


@pytest.mark.parametrize("root", ROOTS)
def test_per_package_opt_out(clean, monkeypatch, root):
    monkeypatch.setenv(guard._PACKAGES[root][0], "0")
    active = guard.install_import_guards()
    assert root not in active
    assert not _active(root)


def test_unrelated_imports_untouched(clean):
    guard.install_import_guards()
    finder = guard._active_finder("deep_gemm")
    assert finder.find_spec("deep_gemm_extra", None) is None
    assert finder.find_spec("json", None) is None


def test_activate_platform_runs_vendor_early_hook(monkeypatch):
    import sglang_fl

    seen = []
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: object() if name.endswith(".early") else None
    )
    monkeypatch.setattr(importlib, "import_module", lambda name: seen.append(name))
    sglang_fl._apply_vendor_early_patches("thead")
    assert seen == ["sglang_fl.dispatch.backends.vendor.thead.early"]


def test_vendor_without_early_module_is_skipped(monkeypatch):
    import sglang_fl

    called = []
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(importlib, "import_module", lambda name: called.append(name))
    sglang_fl._apply_vendor_early_patches("nvidia")
    assert called == []
