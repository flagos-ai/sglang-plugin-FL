"""PPU (thead) PP mem_fraction_static default."""

from types import SimpleNamespace

import pytest

from sglang_fl.dispatch.backends.vendor.thead.runtime_config import (
    apply_thead_runtime_defaults,
)

CFG = {"pp_mem_fraction_static": 0.75}


def _args(pp_size=2, mem=None):
    return SimpleNamespace(pp_size=pp_size, mem_fraction_static=mem)


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("SGLANG_FL_THEAD_PP_MEM_FRACTION", raising=False)


def test_pp_default_applied():
    args = _args()
    apply_thead_runtime_defaults(args, CFG)
    assert args.mem_fraction_static == 0.75


def test_no_pp_untouched():
    args = _args(pp_size=1)
    apply_thead_runtime_defaults(args, CFG)
    assert args.mem_fraction_static is None


def test_user_value_kept():
    args = _args(mem=0.85)
    apply_thead_runtime_defaults(args, CFG)
    assert args.mem_fraction_static == 0.85


def test_env_overrides_config(monkeypatch):
    monkeypatch.setenv("SGLANG_FL_THEAD_PP_MEM_FRACTION", "0.7")
    args = _args()
    apply_thead_runtime_defaults(args, CFG)
    assert args.mem_fraction_static == 0.7


@pytest.mark.parametrize("value", ["auto", "none"])
def test_env_auto_disables(monkeypatch, value):
    monkeypatch.setenv("SGLANG_FL_THEAD_PP_MEM_FRACTION", value)
    args = _args()
    apply_thead_runtime_defaults(args, CFG)
    assert args.mem_fraction_static is None


def test_missing_config_key_keeps_sglang_default():
    args = _args()
    apply_thead_runtime_defaults(args, {})
    assert args.mem_fraction_static is None


@pytest.mark.parametrize("value", ["0", "1", "1.5"])
def test_out_of_range_rejected(monkeypatch, value):
    monkeypatch.setenv("SGLANG_FL_THEAD_PP_MEM_FRACTION", value)
    with pytest.raises(ValueError):
        apply_thead_runtime_defaults(_args(), CFG)


def test_shipped_thead_yaml_has_default():
    from sglang_fl.dispatch.config.utils import load_platform_config

    assert load_platform_config("thead")["pp_mem_fraction_static"] == 0.75
