"""PPU defaults to the triton attention backend (fa3 cannot load on PPU)."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_platform_module(monkeypatch):
    for name in ("sglang", "sglang.srt", "sglang.srt.platforms"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    mixin = ModuleType("sglang.srt.platforms.device_mixin")
    mixin.PlatformEnum = type("PlatformEnum", (), {"OOT": "oot"})
    mixin.DeviceCapability = object
    interface = ModuleType("sglang.srt.platforms.interface")
    interface.SRTPlatform = object
    monkeypatch.setitem(sys.modules, mixin.__name__, mixin)
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    path = Path(__file__).resolve().parents[3] / "sglang_fl/platform.py"
    spec = importlib.util.spec_from_file_location("_thead_platform_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_thead_default_attention_backend_is_triton(monkeypatch):
    module = _load_platform_module(monkeypatch)
    assert module._ATTN_BACKEND_MAP["thead"] == "triton"
