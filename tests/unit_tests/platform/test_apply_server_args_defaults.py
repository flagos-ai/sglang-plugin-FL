from types import SimpleNamespace

import pytest

from sglang_fl.platform import PlatformFL


def _platform(vendor_name: str, device_type: str = "cuda") -> PlatformFL:
    p = PlatformFL.__new__(PlatformFL)
    p._vendor_name = vendor_name
    p._device_type = device_type
    return p


class TestApplyServerArgsDefaults:
    def test_nvidia_skips_even_when_attention_unset(self):
        args = SimpleNamespace(attention_backend=None)
        _platform("nvidia").apply_server_args_defaults(args)
        assert args.attention_backend is None

    def test_nvidia_does_not_override_explicit_backend(self):
        args = SimpleNamespace(attention_backend="triton")
        _platform("nvidia").apply_server_args_defaults(args)
        assert args.attention_backend == "triton"

    def test_iluvatar_defaults_to_triton(self):
        args = SimpleNamespace(attention_backend=None)
        _platform("iluvatar").apply_server_args_defaults(args)
        assert args.attention_backend == "triton"

    def test_iluvatar_respects_explicit_backend(self):
        args = SimpleNamespace(attention_backend="flashinfer")
        _platform("iluvatar").apply_server_args_defaults(args)
        assert args.attention_backend == "flashinfer"

    def test_ascend_defaults_to_ascend(self):
        args = SimpleNamespace(attention_backend=None)
        _platform("ascend", device_type="npu").apply_server_args_defaults(args)
        assert args.attention_backend == "ascend"

    def test_unknown_vendor_falls_back_to_torch_native(self):
        args = SimpleNamespace(attention_backend=None)
        _platform("unknown_vendor", device_type="xpu").apply_server_args_defaults(args)
        assert args.attention_backend == "torch_native"
