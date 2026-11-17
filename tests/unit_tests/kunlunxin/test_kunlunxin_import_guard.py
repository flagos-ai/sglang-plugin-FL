# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for ``early.py`` + ``import_guard.py``.

On cuda-compat Kunlun, SGLang takes its CUDA import branches and imports
optional NVIDIA packages at module level (``flashinfer`` in fp8_utils,
``deep_gemm`` in dsa_backend). Neither is installed, so the guard serves a
placeholder package that fails only when an object is actually called — and
switches the matching SGLang availability flag off so no code path selects it.

Real package roots are environment specific, so the tests register synthetic
probe roots in ``_PACKAGES`` and clean the finders up afterwards.
"""

import ast
import importlib
import os
import sys
import types
from pathlib import Path

import pytest

from sglang_fl.dispatch.backends.vendor.kunlunxin import import_guard

PROBE = "klx_probe_missing_pkg"
OPT_OUT_ENV = "SGLANG_FL_KUNLUNXIN_PROBE_STUB"
ENV_OVERRIDE = "SGLANG_FL_KUNLUNXIN_PROBE_FLAG"
SECONDARY = "klx_probe_installed_pkg"


@pytest.fixture
def probe_roots(monkeypatch):
    """Register synthetic roots and guarantee global state is restored."""
    monkeypatch.setitem(
        import_guard._PACKAGES, PROBE, (OPT_OUT_ENV, {ENV_OVERRIDE: "false"})
    )
    monkeypatch.setitem(
        import_guard._PACKAGES, SECONDARY, (OPT_OUT_ENV, {ENV_OVERRIDE: "false"})
    )
    monkeypatch.delenv(OPT_OUT_ENV, raising=False)
    saved_override = os.environ.get(ENV_OVERRIDE)
    try:
        yield
    finally:
        if saved_override is None:
            os.environ.pop(ENV_OVERRIDE, None)
        else:
            os.environ[ENV_OVERRIDE] = saved_override
        for root in (PROBE, SECONDARY):
            _drop_finders(root)
            _drop_modules(root)


def _drop_finders(root):
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not (
            isinstance(finder, import_guard._PlaceholderFinder) and finder.root == root
        )
    ]


def _drop_modules(root):
    for name in [n for n in sys.modules if n == root or n.startswith(root + ".")]:
        del sys.modules[name]


def _finders(root):
    return [
        finder
        for finder in sys.meta_path
        if isinstance(finder, import_guard._PlaceholderFinder) and finder.root == root
    ]


def test_missing_package_becomes_a_call_failing_placeholder(probe_roots, monkeypatch):
    monkeypatch.delenv(ENV_OVERRIDE, raising=False)

    assert import_guard.install_placeholder(PROBE) is True
    assert len(_finders(PROBE)) == 1
    assert os.environ[ENV_OVERRIDE] == "false"

    module = importlib.import_module(PROBE)
    assert module.__sglang_fl_placeholder__ is True
    assert module.__path__ == []

    attribute = module.some_kernel
    assert repr(attribute) == "<unavailable klx_probe_missing_pkg.some_kernel>"
    with pytest.raises(RuntimeError) as caught:
        attribute(1, 2, key="value")
    message = str(caught.value)
    assert PROBE in message and "not installed" in message and "placeholder" in message


def test_submodules_are_served_by_the_same_placeholder(probe_roots):
    import_guard.install_placeholder(PROBE)

    submodule = importlib.import_module(f"{PROBE}.sub.module")
    assert submodule.__sglang_fl_placeholder__ is True
    with pytest.raises(RuntimeError, match=PROBE):
        submodule.function()


def test_dunder_attributes_stay_honest(probe_roots):
    import_guard.install_placeholder(PROBE)
    module = importlib.import_module(PROBE)

    with pytest.raises(AttributeError):
        module.__deepcopy__
    with pytest.raises(AttributeError):
        module.some_attr.__wrapped__


def test_placeholder_objects_can_be_used_as_base_classes(probe_roots):
    import_guard.install_placeholder(PROBE)
    module = importlib.import_module(PROBE)

    class Derived(module.Base):
        pass

    assert Derived.__mro__[1] is object


def test_opt_out_env_disables_the_placeholder(probe_roots, monkeypatch):
    monkeypatch.setenv(OPT_OUT_ENV, "0")

    assert import_guard.install_placeholder(PROBE) is False
    assert _finders(PROBE) == []


def test_installed_package_always_wins(probe_roots, tmp_path, monkeypatch):
    package = tmp_path / SECONDARY
    package.mkdir()
    (package / "__init__.py").write_text("REAL = True\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    assert import_guard.install_placeholder(SECONDARY) is False
    assert _finders(SECONDARY) == []
    assert importlib.import_module(SECONDARY).REAL is True


def test_install_is_idempotent(probe_roots):
    assert import_guard.install_placeholder(PROBE) is True
    finder = _finders(PROBE)[0]

    assert import_guard.install_placeholder(PROBE) is True
    assert _finders(PROBE) == [finder]


def test_install_import_guards_reports_only_active_roots(
    probe_roots, tmp_path, monkeypatch
):
    package = tmp_path / SECONDARY
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    active = import_guard.install_import_guards()

    assert PROBE in active
    assert SECONDARY not in active


def test_early_module_installs_the_guards(probe_roots, monkeypatch):
    early_name = "sglang_fl.dispatch.backends.vendor.kunlunxin.early"
    monkeypatch.delitem(sys.modules, early_name, raising=False)

    early = importlib.import_module(early_name)

    assert _finders(PROBE)
    assert hasattr(early, "install_import_guards")


def test_early_module_does_not_import_sglang_srt():
    """``early.py`` runs during platform discovery, before sglang.srt.layers."""
    early_path = Path(import_guard.__file__).with_name("early.py")
    tree = ast.parse(early_path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not [name for name in imported if name.startswith("sglang.srt")]


def test_placeholder_module_type_is_a_real_module():
    """``types.ModuleType`` matters: some libraries inspect module internals."""
    module = import_guard._PlaceholderModule("some.name")
    assert isinstance(module, types.ModuleType)
