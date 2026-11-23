"""Hygon plugin-owned placeholders for missing SGLang kernel wheels."""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GUARD_PATH = REPO_ROOT / "sglang_fl/dispatch/backends/vendor/hygon/import_guard.py"
_SPEC = importlib.util.spec_from_file_location("hygon_import_guard_test", GUARD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
guard = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(guard)
ROOTS = list(guard._PACKAGES)
_REAL_INSTALLED = guard._installed


@pytest.fixture()
def clean(monkeypatch):
    monkeypatch.setattr(
        sys,
        "meta_path",
        [f for f in sys.meta_path if not isinstance(f, guard._PlaceholderFinder)],
    )
    for root in ROOTS:
        for name in [m for m in sys.modules if m == root or m.startswith(root + ".")]:
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.delenv(guard._PACKAGES[root], raising=False)
    monkeypatch.delenv(guard._SKIP_VERSION_CHECK, raising=False)
    monkeypatch.setattr(guard, "_installed", lambda root: False)
    yield
    for root in ROOTS:
        for name in [m for m in sys.modules if m == root or m.startswith(root + ".")]:
            sys.modules.pop(name, None)


def activate_placeholders_first(monkeypatch):
    guard.install_import_guards()
    placeholders = [f for f in sys.meta_path if isinstance(f, guard._PlaceholderFinder)]
    others = [f for f in sys.meta_path if not isinstance(f, guard._PlaceholderFinder)]
    monkeypatch.setattr(sys, "meta_path", placeholders + others)


def test_exact_semantics(clean, monkeypatch):
    activate_placeholders_first(monkeypatch)

    import flashinfer
    import flashinfer.sampling
    import sgl_kernel
    from sgl_kernel.kvcacheio import save_kv_cache
    from sgl_kernel.utils import is_arch_support_pdl

    assert is_arch_support_pdl() is False
    assert sgl_kernel.__sglang_fl_placeholder__ is True
    assert flashinfer.__sglang_fl_placeholder__ is True
    assert os.environ[guard._SKIP_VERSION_CHECK] == "1"
    with pytest.raises(RuntimeError, match="sgl_kernel is not installed on HCU"):
        save_kv_cache()
    with pytest.raises(RuntimeError, match="flashinfer is not installed on HCU"):
        flashinfer.sampling.top_k_renorm_prob()


def test_scalar_type_preserves_real_import_failure(clean, monkeypatch):
    activate_placeholders_first(monkeypatch)
    assert importlib.util.find_spec("sgl_kernel.scalar_type") is None
    with pytest.raises(ModuleNotFoundError, match="sgl_kernel.scalar_type"):
        importlib.import_module("sgl_kernel.scalar_type")


def test_import_time_subclassing_is_supported(clean, monkeypatch):
    activate_placeholders_first(monkeypatch)
    import flashinfer.fused_moe.core as core

    class Derived(core.SomeBase):
        pass

    assert Derived.__mro__ == (Derived, object)


def test_install_is_idempotent(clean):
    assert guard.install_import_guards() == ROOTS
    assert guard.install_import_guards() == ROOTS
    for root in ROOTS:
        assert (
            sum(
                isinstance(f, guard._PlaceholderFinder) and f.root == root
                for f in sys.meta_path
            )
            == 1
        )


@pytest.mark.parametrize("root", ROOTS)
def test_installed_real_package_wins(clean, monkeypatch, root):
    real = types.ModuleType(root)
    real.__path__ = []

    class RealFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path, target=None):
            if fullname == root:
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
            return None

        def create_module(self, spec):
            return real

        def exec_module(self, module):
            pass

    monkeypatch.setattr(sys, "meta_path", [RealFinder()] + sys.meta_path)
    monkeypatch.setattr(guard, "_installed", _REAL_INSTALLED)
    assert guard.install_placeholder(root) is False
    assert guard._active_finder(root) is None
    assert importlib.import_module(root) is real
    if root == "sgl_kernel":
        assert guard._SKIP_VERSION_CHECK not in os.environ


@pytest.mark.parametrize("root", ROOTS)
def test_per_package_opt_out(clean, monkeypatch, root):
    monkeypatch.setenv(guard._PACKAGES[root], "0")
    assert root not in guard.install_import_guards()
    assert guard._active_finder(root) is None


def test_unrelated_imports_are_untouched(clean):
    guard.install_import_guards()
    finder = guard._active_finder("sgl_kernel")
    assert finder.find_spec("sgl_kernel_extra", None) is None
    assert finder.find_spec("json", None) is None


def test_hygon_early_hook_is_selected(monkeypatch):
    pytest.importorskip("torch")
    import sglang_fl

    seen = []
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name.startswith("sglang_fl.dispatch") else None,
    )
    monkeypatch.setattr(importlib, "import_module", lambda name: seen.append(name))
    sglang_fl._apply_vendor_early_patches("hygon")
    assert seen == ["sglang_fl.dispatch.backends.vendor.hygon.early"]


@pytest.mark.parametrize("vendor", ["nvidia", "ascend", "no_such_vendor"])
def test_non_hygon_vendor_does_not_load_hygon_guard(monkeypatch, vendor):
    pytest.importorskip("torch")
    import sglang_fl

    called = []
    real_import = importlib.import_module

    def tracking_import(name, *args, **kwargs):
        if name.endswith("hygon.early"):
            called.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", tracking_import)
    sglang_fl._apply_vendor_early_patches(vendor)
    assert called == []


def test_parent_child_grandchild_each_activate_the_guard():
    code = r"""
import importlib
import json
import os
import subprocess
import sys

spec = importlib.util.spec_from_file_location(
    "hygon_import_guard_child", os.environ["HYGON_GUARD_PATH"]
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

guard._installed = lambda root: False
active = guard.install_import_guards()
from sgl_kernel.utils import is_arch_support_pdl
try:
    from sgl_kernel.kvcacheio import save_kv_cache
    save_kv_cache()
except RuntimeError as error:
    loud = "sgl_kernel is not installed on HCU" in str(error)
else:
    loud = False

depth = int(os.environ.get("HYGON_GUARD_DEPTH", "0"))
result = {
    "depth": depth,
    "active": active,
    "pdl": is_arch_support_pdl(),
    "scalar_type": importlib.util.find_spec("sgl_kernel.scalar_type") is not None,
    "loud": loud,
}
if depth < 2:
    env = os.environ.copy()
    env["HYGON_GUARD_DEPTH"] = str(depth + 1)
    child = subprocess.run(
        [sys.executable, "-c", os.environ["HYGON_GUARD_CODE"]],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    result["child"] = json.loads(child.stdout)
print(json.dumps(result))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["HYGON_GUARD_PATH"] = str(GUARD_PATH)
    env["HYGON_GUARD_CODE"] = code
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    for depth in range(3):
        assert result == {
            **result,
            "depth": depth,
            "active": ROOTS,
            "pdl": False,
            "scalar_type": False,
            "loud": True,
        }
        result = result.get("child", {})
