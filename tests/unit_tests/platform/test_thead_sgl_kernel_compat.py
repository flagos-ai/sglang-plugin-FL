"""PPU sgl_kernel needs T-Head-only SGLang symbols; the thead patch provides them."""

import sys
from types import ModuleType

import pytest

from sglang_fl.dispatch.backends.vendor.thead.patches import sgl_kernel_compat as compat


def _fake_utils(monkeypatch, **attrs):
    mods = {}
    for name in compat._UTILS_MODULES:
        mod = ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, name, mod)
        mods[name] = mod
    return mods


def test_is_ppu_added_when_missing(monkeypatch):
    mods = _fake_utils(monkeypatch)
    compat._patch_is_ppu()
    for mod in mods.values():
        assert mod.is_ppu is compat.is_ppu
    assert mods["sglang.srt.utils"].is_ppu() is True


def test_existing_is_ppu_is_kept(monkeypatch):
    def vendor_is_ppu():
        return True

    mods = _fake_utils(monkeypatch, is_ppu=vendor_is_ppu)
    compat._patch_is_ppu()
    for mod in mods.values():
        assert mod.is_ppu is vendor_is_ppu


def test_missing_sglang_is_ignored(monkeypatch):
    for name in compat._UTILS_MODULES:
        monkeypatch.setitem(sys.modules, name, None)  # import -> ImportError
    monkeypatch.setitem(sys.modules, "sglang.srt.environ", None)
    compat.patch_sgl_kernel_compat()


def test_nvtx_envs_added_with_false_default(monkeypatch):
    environ = pytest.importorskip("sglang.srt.environ")
    for name in compat._ENV_BOOLS:
        if name in vars(environ.Envs):
            monkeypatch.delattr(environ.Envs, name)
        monkeypatch.delenv(name, raising=False)
    compat._patch_envs()
    try:
        for name in compat._ENV_BOOLS:
            field = getattr(environ.envs, name)
            assert field.name == name
            assert field.get() is False
        monkeypatch.setenv("SGLANG_PROFILE_NVTX", "1")
        assert environ.envs.SGLANG_PROFILE_NVTX.get() is True
    finally:
        for name in compat._ENV_BOOLS:
            if name in vars(environ.Envs):
                delattr(environ.Envs, name)
