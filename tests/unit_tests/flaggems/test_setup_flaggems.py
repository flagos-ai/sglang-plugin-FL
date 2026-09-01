from __future__ import annotations

import logging
import sys
from types import ModuleType

import pytest


@pytest.fixture
def sglang_fl_module():
    import sglang_fl

    return sglang_fl


@pytest.fixture
def fake_flag_gems(monkeypatch):
    module = ModuleType("flag_gems")
    calls = []

    def enable(**kwargs):
        calls.append(("enable", kwargs))

    def only_enable(**kwargs):
        calls.append(("only_enable", kwargs))

    module.enable = enable
    module.only_enable = only_enable
    module.calls = calls
    monkeypatch.setitem(sys.modules, "flag_gems", module)
    return module


@pytest.fixture(autouse=True)
def clean_flaggems_env(monkeypatch):
    for key in (
        "USE_FLAGGEMS",
        "SGLANG_FL_FLAGOS_WHITELIST",
        "SGLANG_FL_FLAGOS_BLACKLIST",
        "SGLANG_FLAGGEMS_RECORD",
        "SGLANG_FLAGGEMS_LOG_PATH",
        "SGLANG_FLAGGEMS_LOG_ONCE",
        "SGLANG_FL_CONFIG",
        "SGLANG_FL_PLATFORM",
    ):
        monkeypatch.delenv(key, raising=False)


def test_use_flaggems_zero_does_not_import_or_enable(
    monkeypatch,
    sglang_fl_module,
) -> None:
    monkeypatch.setenv("USE_FLAGGEMS", "0")
    monkeypatch.delitem(sys.modules, "flag_gems", raising=False)

    sglang_fl_module._setup_flaggems({})

    assert "flag_gems" not in sys.modules


def test_no_whitelist_or_blacklist_calls_enable(
    monkeypatch,
    sglang_fl_module,
    fake_flag_gems,
) -> None:
    monkeypatch.setenv("USE_FLAGGEMS", "1")

    sglang_fl_module._setup_flaggems({})

    assert fake_flag_gems.calls == [
        ("enable", {"record": False, "once": True}),
    ]


def test_flagos_whitelist_calls_only_enable(
    monkeypatch,
    sglang_fl_module,
    fake_flag_gems,
) -> None:
    monkeypatch.setenv("SGLANG_FL_FLAGOS_WHITELIST", "silu, rms_norm")

    sglang_fl_module._setup_flaggems({})

    assert fake_flag_gems.calls == [
        (
            "only_enable",
            {
                "include": ["silu", "rms_norm"],
                "record": False,
                "once": True,
            },
        ),
    ]


def test_flagos_blacklist_calls_enable_with_unused(
    monkeypatch,
    sglang_fl_module,
    fake_flag_gems,
) -> None:
    monkeypatch.setenv("SGLANG_FL_FLAGOS_BLACKLIST", "count_nonzero, var_mean")

    sglang_fl_module._setup_flaggems({})

    assert fake_flag_gems.calls == [
        (
            "enable",
            {
                "unused": ["count_nonzero", "var_mean"],
                "record": False,
                "once": True,
            },
        ),
    ]


def test_flagos_whitelist_and_blacklist_are_mutually_exclusive(
    monkeypatch,
    sglang_fl_module,
    fake_flag_gems,
) -> None:
    monkeypatch.setenv("SGLANG_FL_FLAGOS_WHITELIST", "silu")
    monkeypatch.setenv("SGLANG_FL_FLAGOS_BLACKLIST", "rms_norm")

    with pytest.raises(ValueError, match="Cannot set both"):
        sglang_fl_module._setup_flaggems({})

    assert fake_flag_gems.calls == []


def test_flagos_lists_strip_spaces_and_empty_items(
    monkeypatch,
    sglang_fl_module,
    fake_flag_gems,
) -> None:
    monkeypatch.setenv("SGLANG_FL_FLAGOS_BLACKLIST", " silu , , rms_norm , ")

    sglang_fl_module._setup_flaggems({})

    assert fake_flag_gems.calls[0] == (
        "enable",
        {
            "unused": ["silu", "rms_norm"],
            "record": False,
            "once": True,
        },
    )


def test_flaggems_record_and_log_path_are_forwarded(
    monkeypatch,
    sglang_fl_module,
    fake_flag_gems,
) -> None:
    monkeypatch.setenv("SGLANG_FLAGGEMS_RECORD", "1")
    monkeypatch.setenv("SGLANG_FLAGGEMS_LOG_PATH", "/tmp/flaggems.log")
    monkeypatch.setenv("SGLANG_FLAGGEMS_LOG_ONCE", "0")

    sglang_fl_module._setup_flaggems()

    assert fake_flag_gems.calls == [
        (
            "enable",
            {
                "record": True,
                "once": False,
                "path": "/tmp/flaggems.log",
            },
        ),
    ]


def test_flaggems_record_installs_aten_only_filter(
    sglang_fl_module,
    fake_flag_gems,
) -> None:
    logger = logging.getLogger("flag_gems")
    handler = logging.StreamHandler()
    handler._flaggems_owned = True
    old_handlers = list(logger.handlers)
    logger.handlers = [handler]
    try:
        sglang_fl_module._setup_flaggems(
            {
                "flaggems_record": True,
                "flaggems_log_path": "",
                "flagos_blacklist": [],
            }
        )
        assert handler.filters
        filter_ = handler.filters[-1]
        assert filter_.filter(logging.LogRecord(
            "flag_gems.ops.add",
            logging.INFO,
            "",
            0,
            "",
            (),
            None,
        ))
        assert not filter_.filter(logging.LogRecord(
            "flag_gems.modules.activation",
            logging.INFO,
            "",
            0,
            "",
            (),
            None,
        ))
    finally:
        logger.handlers = old_handlers


def test_yaml_flagos_blacklist_is_default_when_env_blacklist_absent(
    sglang_fl_module,
    fake_flag_gems,
) -> None:
    sglang_fl_module._setup_flaggems(
        {
            "flaggems_record": False,
            "flaggems_log_path": "",
            "flagos_blacklist": ["count_nonzero"],
        }
    )

    assert fake_flag_gems.calls == [
        (
            "enable",
            {
                "unused": ["count_nonzero"],
                "record": False,
                "once": True,
            },
        ),
    ]


def test_env_blacklist_overrides_yaml_flagos_blacklist(
    monkeypatch,
    sglang_fl_module,
    fake_flag_gems,
) -> None:
    monkeypatch.setenv("SGLANG_FL_FLAGOS_BLACKLIST", "silu")

    sglang_fl_module._setup_flaggems(
        {
            "flaggems_record": False,
            "flaggems_log_path": "",
            "flagos_blacklist": ["count_nonzero"],
        }
    )

    assert fake_flag_gems.calls == [
        (
            "enable",
            {
                "unused": ["silu"],
                "record": False,
                "once": True,
            },
        ),
    ]


def test_build_config_reads_flaggems_env(
    monkeypatch,
    sglang_fl_module,
) -> None:
    import sglang_fl.dispatch.config as config_module

    monkeypatch.setattr(config_module, "get_effective_config", lambda: {})
    monkeypatch.setenv("SGLANG_FLAGGEMS_RECORD", "true")
    monkeypatch.setenv("SGLANG_FLAGGEMS_LOG_PATH", "/tmp/fg.log")

    config = sglang_fl_module._build_config()

    assert config["flaggems_record"] is True
    assert config["flaggems_log_path"] == "/tmp/fg.log"


def test_build_config_uses_yaml_flagos_blacklist_when_env_absent(
    monkeypatch,
    sglang_fl_module,
) -> None:
    import sglang_fl.dispatch.config as config_module

    monkeypatch.setattr(
        config_module,
        "get_effective_config",
        lambda: {"flagos_blacklist": ["yaml_op"]},
    )

    config = sglang_fl_module._build_config()

    assert config["flagos_blacklist"] == ["yaml_op"]


def test_build_config_env_flagos_blacklist_overrides_yaml(
    monkeypatch,
    sglang_fl_module,
) -> None:
    import sglang_fl.dispatch.config as config_module

    monkeypatch.setattr(
        config_module,
        "get_effective_config",
        lambda: {"flagos_blacklist": ["yaml_op"]},
    )
    monkeypatch.setenv("SGLANG_FL_FLAGOS_BLACKLIST", " env_op , other_op ")

    config = sglang_fl_module._build_config()

    assert config["flagos_blacklist"] == ["env_op", "other_op"]
