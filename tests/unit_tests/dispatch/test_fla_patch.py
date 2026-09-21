import builtins
import logging

from sglang_fl.dispatch.fla_patch import patch_fla_functions


def test_missing_optional_fla_modules_are_skipped(monkeypatch, caplog):
    real_import = builtins.__import__

    def missing_fla(name, *args, **kwargs):
        if name.startswith("sglang.srt.layers.attention.fla"):
            exc = ModuleNotFoundError(name)
            exc.name = "sglang.srt.layers.attention.fla"
            raise exc
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_fla)
    with caplog.at_level(logging.INFO):
        assert patch_fla_functions() is None
    assert "FLA modules are absent" in caplog.text
