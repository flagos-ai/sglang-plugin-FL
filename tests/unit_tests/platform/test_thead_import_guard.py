"""Tests for the PPU (thead) import placeholders (flashinfer, deep_gemm,
sgl_kernel — the last with its exact semantic contract)."""

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
_REAL_INSTALLED = (
    guard._installed
)  # the fixture stubs it out; one test needs the real one


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


def test_guards_all_packages_and_forces_availability_off(clean):
    assert guard.install_import_guards() == ["flashinfer", "deep_gemm", "sgl_kernel"]
    assert os.environ["SGLANG_IS_FLASHINFER_AVAILABLE"] == "false"
    assert os.environ["SGLANG_ENABLE_JIT_DEEPGEMM"] == "false"
    # The sgl_kernel placeholder serves no dist-info, so SGLang's
    # wheel-version probe must be skipped while it is active.
    assert os.environ["SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"] == "true"


def test_module_level_import_patterns(clean, monkeypatch):
    guard.install_import_guards()
    # The guard appends its finders so an installed package always wins. CI hosts
    # (CUDA/Hygon/MUSA images) do ship flashinfer / deep_gemm, so put the
    # placeholders first to exercise them as on a PPU host where both are absent.
    placeholders = [f for f in sys.meta_path if isinstance(f, guard._PlaceholderFinder)]
    others = [f for f in sys.meta_path if not isinstance(f, guard._PlaceholderFinder)]
    monkeypatch.setattr(sys, "meta_path", placeholders + others)

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


# --- sgl_kernel semantic placeholder ----------------------------------------
#
# The two-wheel candidate image ships NO sglang-kernel distribution; the
# placeholder answers the srt control path's real queries and stays silent
# otherwise. These tests pin the exact contract:
#   * is_arch_support_pdl() -> False (the real control-flow answer);
#   * scalar_type does not exist, so SGLang's own fallback selects the
#     non-kernel path (no usable-looking fake);
#   * every other member fails loudly on the first real call (no
#     flash_mla/flash_attn fakes, no torch.ops registrations);
#   * a real sgl_kernel beats the placeholder, both through the
#     _installed() pre-check and through meta_path order — which is what
#     makes the kernel-wheel fallback safe with no code change.


def _placeholders_first(monkeypatch):
    """Put the placeholder finders ahead of the real path finder.

    CI images (CUDA/Hygon/MUSA) ship real sgl_kernel/flashinfer/deep_gemm;
    on a PPU host none is importable and the appended placeholder wins.
    Reordering reproduces the PPU host exactly (the same trick the
    flashinfer/deep_gemm import-pattern test uses).
    """
    placeholders = [f for f in sys.meta_path if isinstance(f, guard._PlaceholderFinder)]
    others = [f for f in sys.meta_path if not isinstance(f, guard._PlaceholderFinder)]
    monkeypatch.setattr(sys, "meta_path", placeholders + others)


def test_sgl_kernel_pdl_query_answers_real_false(clean, monkeypatch):
    guard.install_import_guards()
    _placeholders_first(monkeypatch)
    from sgl_kernel.utils import is_arch_support_pdl

    assert is_arch_support_pdl() is False


def test_sgl_kernel_scalar_type_is_absent(clean, monkeypatch):
    guard.install_import_guards()
    _placeholders_first(monkeypatch)
    import sgl_kernel

    assert not hasattr(sgl_kernel, "scalar_type")
    with pytest.raises(ImportError):
        from sgl_kernel import scalar_type  # noqa: F401
    # The submodule-import fallback (what `from X import Y` tries after the
    # AttributeError) must refuse too — no placeholder module may exist for
    # an _ABSENT member.
    with pytest.raises(ModuleNotFoundError):
        import sgl_kernel.scalar_type  # noqa: F401


def test_sgl_kernel_unknown_members_fail_loudly(clean, monkeypatch):
    guard.install_import_guards()
    _placeholders_first(monkeypatch)
    import sgl_kernel
    import sgl_kernel.utils

    # Submodule members: importable, loud on the first real call.
    with pytest.raises(RuntimeError, match="sgl_kernel.utils.any_op is unavailable"):
        sgl_kernel.utils.any_op()
    # Root members such as the kernel entry points: never faked.
    with pytest.raises(RuntimeError, match="sgl_kernel.flash_mla is unavailable"):
        sgl_kernel.flash_mla()
    with pytest.raises(RuntimeError, match="sgl_kernel.flash_attn is unavailable"):
        sgl_kernel.flash_attn()


def test_torch_ops_sgl_kernel_not_faked(clean, monkeypatch):
    # sglang's own probe (srt/utils/common.py): with no registered op the
    # native fallback path is selected. The placeholder cannot fake torch
    # ops and must not try.
    torch = pytest.importorskip("torch")
    guard.install_import_guards()
    _placeholders_first(monkeypatch)
    assert not hasattr(torch.ops.sgl_kernel, "convert_weight_packed")


def test_installed_sgl_kernel_disables_placeholder(clean, monkeypatch):
    # The pre-check: a real sgl_kernel importable from meta_path means the
    # placeholder never activates for that root (others stay guarded).
    monkeypatch.setattr(guard, "_installed", lambda root: root == "sgl_kernel")
    active = guard.install_import_guards()
    assert "sgl_kernel" not in active
    assert not _active("sgl_kernel")
    assert _active("flashinfer")


def test_real_sgl_kernel_beats_placeholder(clean, monkeypatch):
    # meta_path order wins: a finder serving a real module ahead of the
    # placeholder resolves the real package (mirrors how an installed
    # distribution shadows an appended finder).
    real = types.ModuleType("sgl_kernel")
    real.__path__ = []
    real.sentinel = "real"

    class _RealFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path, target=None):
            if fullname == "sgl_kernel":
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
            return None

        def create_module(self, spec):
            return real if spec.name == "sgl_kernel" else None

        def exec_module(self, module):
            pass

    guard.install_import_guards()
    placeholders = [f for f in sys.meta_path if isinstance(f, guard._PlaceholderFinder)]
    others = [f for f in sys.meta_path if not isinstance(f, guard._PlaceholderFinder)]
    monkeypatch.setattr(sys, "meta_path", [_RealFinder()] + placeholders + others)

    import sgl_kernel

    assert getattr(sgl_kernel, "sentinel", None) == "real"
    assert not getattr(sgl_kernel, "__sglang_fl_placeholder__", False)


def test_unrelated_imports_untouched(clean):
    guard.install_import_guards()
    finder = guard._active_finder("deep_gemm")
    assert finder.find_spec("deep_gemm_extra", None) is None
    assert finder.find_spec("json", None) is None


def test_activate_platform_runs_vendor_early_hook(monkeypatch):
    import sglang_fl

    seen = []
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        # The vendor package and its early module both resolve.
        lambda name: object() if name.startswith("sglang_fl.dispatch") else None,
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


@pytest.mark.parametrize("vendor", ["nvidia", "no_such_vendor"])
def test_vendor_without_vendor_package_is_skipped(monkeypatch, vendor):
    # Real import machinery: find_spec raises ModuleNotFoundError (not None) when
    # vendor/<vendor>/ itself is missing, which is the case for nvidia (vendor/cuda/).
    import sglang_fl

    called = []
    real_import = importlib.import_module

    def tracking_import(name, *args, **kwargs):
        if name.endswith(".early"):
            called.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", tracking_import)
    sglang_fl._apply_vendor_early_patches(vendor)
    assert called == []


# nvidia has no vendor/ package; ascend has one but no early.py.
@pytest.mark.parametrize("vendor", ["nvidia", "ascend"])
def test_activate_platform_succeeds_without_vendor_early_module(monkeypatch, vendor):
    import sglang_fl
    from sglang_fl import utils as fl_utils

    monkeypatch.setattr(
        fl_utils,
        "get_device_info",
        lambda: types.SimpleNamespace(vendor_name=vendor, device_type="test"),
    )
    assert sglang_fl.activate_platform() == "sglang_fl.platform:PlatformFL"
