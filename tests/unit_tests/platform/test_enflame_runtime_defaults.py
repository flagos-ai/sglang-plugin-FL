"""CPU checks for GCU platform defaults and ordinary decode graph capability."""
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


@pytest.fixture
def platform_class(monkeypatch):
    # Isolate SGLang interfaces so these unit tests never discover real hardware.
    for name in ('sglang', 'sglang.srt', 'sglang.srt.platforms'):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    mixin = ModuleType('sglang.srt.platforms.device_mixin')
    mixin.PlatformEnum = SimpleNamespace(OOT='oot')
    mixin.DeviceCapability = object
    interface = ModuleType('sglang.srt.platforms.interface')
    interface.SRTPlatform = object
    monkeypatch.setitem(sys.modules, mixin.__name__, mixin)
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    path = Path(__file__).resolve().parents[3] / 'sglang_fl/platform.py'
    spec = importlib.util.spec_from_file_location('enflame_platform_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PlatformFL


def test_gcu_runtime_defaults_preserve_graph_selection(platform_class):
    platform = platform_class.__new__(platform_class)
    platform._vendor_name, platform._device_type = 'enflame', 'gcu'
    args = SimpleNamespace(cuda_graph_backend_prefill=None, cuda_graph_backend_decode='full')
    platform.apply_server_args_defaults(args)
    assert (args.device, args.attention_backend, args.mm_attention_backend) == ('gcu', 'fa3', 'fa3')
    assert args.page_size == 64
    assert args.watchdog_timeout == 100000
    assert args.disable_radix_cache is True
    assert args.cuda_graph_backend_prefill is None
    assert args.cuda_graph_backend_decode == 'full'


def test_nvidia_defaults_are_untouched(platform_class):
    platform = platform_class.__new__(platform_class)
    platform._vendor_name, platform._device_type = 'nvidia', 'cuda'
    args = SimpleNamespace(device='cuda', attention_backend='chosen', page_size=16)
    before = vars(args).copy()
    platform.apply_server_args_defaults(args)
    assert vars(args) == before


def test_gcu_graph_capability(platform_class):
    platform = platform_class.__new__(platform_class)
    platform._vendor_name, platform._device_type = 'enflame', 'gcu'
    assert platform.support_cuda_graph() is True
    assert platform.support_piecewise_cuda_graph() is False
