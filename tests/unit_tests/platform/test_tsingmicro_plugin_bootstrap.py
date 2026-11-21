# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TXDA model-config plugin bootstrap (sglang.bench_one_batch workers); no device work."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import sglang_fl

_MODULE_PATH = (
    Path(sglang_fl.__file__).parent
    / "dispatch/backends/vendor/tsingmicro/plugin_bootstrap.py"
)


def _owned_hooks(patched):
    return SimpleNamespace(
        _hooks={
            "owned.target": [
                (None, lambda: None, SimpleNamespace(plugin_name="sglang_fl"))
            ],
            "other.target": [
                (None, lambda: None, SimpleNamespace(plugin_name="another_plugin"))
            ],
        },
        _patched=patched,
    )


def _install(monkeypatch, original, loader, active, hooks=None):
    class ModelConfig:
        from_server_args = staticmethod(original)

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.configs.model_config",
        SimpleNamespace(ModelConfig=ModelConfig),
    )
    monkeypatch.setitem(
        sys.modules, "sglang.srt.plugins", SimpleNamespace(load_plugins=loader)
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.plugins.hook_registry",
        SimpleNamespace(HookRegistry=hooks or _owned_hooks({"owned.target"})),
    )
    monkeypatch.setattr(sglang_fl, "is_plugin_active", lambda: active[0])
    # Fresh module state per test.
    spec = importlib.util.spec_from_file_location(
        "sglang_fl.dispatch.backends.vendor.tsingmicro._test_plugin_bootstrap",
        _MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install_model_config_bootstrap()
    return ModelConfig, module


def test_worker_without_plugin_loads_it_once_before_model_config(monkeypatch) -> None:
    calls, active, result = [], [False], object()

    def loader():
        calls.append("load_plugins")
        active[0] = True

    def original(args, *, is_draft_model=False):
        calls.append(("model_config", is_draft_model))
        return result

    cls, module = _install(monkeypatch, original, loader, active)
    wrapped = cls.__dict__["from_server_args"]
    assert isinstance(wrapped, staticmethod)
    module.install_model_config_bootstrap()  # idempotent
    assert cls.__dict__["from_server_args"] is wrapped
    assert calls == []  # nothing loads at install time
    for _ in range(2):
        assert cls.from_server_args(object(), is_draft_model=True) is result
    assert calls == ["load_plugins", ("model_config", True), ("model_config", True)]


def test_active_engine_path_passes_through(monkeypatch) -> None:
    calls, result = [], object()

    def unexpected_loader():
        raise AssertionError("an active Engine must not reenter plugin loading")

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    cls, _ = _install(monkeypatch, original, unexpected_loader, [True])
    arg = object()
    assert cls.from_server_args(arg, model_path="draft") is result
    assert calls == [((arg,), {"model_path": "draft"})]


@pytest.mark.parametrize("failure", ["inactive", "import", "hook", "swallowed_hook"])
def test_incomplete_plugin_is_rejected_and_stays_failed(monkeypatch, failure) -> None:
    calls, active = [], [False]

    def loader():
        calls.append("load_plugins")
        if failure == "import":
            raise ImportError("missing runtime dependency")
        if failure == "hook":
            active[0] = True
            raise RuntimeError("hook application failed")
        if failure == "swallowed_hook":
            active[0] = True  # active, but our hook was never applied

    def original(*args, **kwargs):
        calls.append("model_config")

    cls, _ = _install(monkeypatch, original, loader, active, _owned_hooks(set()))
    with pytest.raises(ImportError if failure == "import" else RuntimeError):
        cls.from_server_args(object())
    with pytest.raises(RuntimeError, match="previously failed"):
        cls.from_server_args(object())
    assert calls == ["load_plugins"]
