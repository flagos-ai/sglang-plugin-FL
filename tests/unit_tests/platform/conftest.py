# Shared fixtures for platform-layer unit tests.

import sys
import types

import pytest


@pytest.fixture
def mock_device_detector(monkeypatch):
    """Factory fixture that stubs ``flag_gems.runtime.backend.device.DeviceDetector``.

    Usage:
        mock_device_detector("fakevendor")                  # success path
        mock_device_detector(raise_exc=RuntimeError("..."))  # failure path

    The stub goes into ``sys.modules`` so the production ``from
    flag_gems.runtime.backend.device import DeviceDetector`` hits it without
    needing flag_gems installed.
    """

    def _make(vendor_name=None, raise_exc=None):
        fake_mod = types.ModuleType("flag_gems.runtime.backend.device")
        if raise_exc is not None:
            def _detector_fail(*args, **kwargs):
                raise raise_exc
            fake_mod.DeviceDetector = _detector_fail
        else:
            class FakeDetector:
                def __init__(self):
                    self.vendor_name = vendor_name
            fake_mod.DeviceDetector = FakeDetector

        for name in ("flag_gems", "flag_gems.runtime", "flag_gems.runtime.backend"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(
            sys.modules, "flag_gems.runtime.backend.device", fake_mod
        )

    return _make


@pytest.fixture
def inject_vendor_module(monkeypatch):
    """Factory fixture that injects a fake submodule under
    ``sglang_fl.dispatch.backends.vendor.<vendor>``.

    Usage:
        inject_vendor_module("fakevendor", "register_platform")
        inject_vendor_module("fakevendor", "patch")

    The injected module lives in ``sys.modules`` so ``importlib.import_module``
    returns it directly without touching disk. The fixture cleans up on teardown.
    """

    def _make(vendor_name, submodule):
        pkg_path = f"sglang_fl.dispatch.backends.vendor.{vendor_name}"
        if pkg_path not in sys.modules:
            monkeypatch.setitem(sys.modules, pkg_path, types.ModuleType(pkg_path))

        full = f"{pkg_path}.{submodule}"
        mod = types.ModuleType(full)
        monkeypatch.setitem(sys.modules, full, mod)
        return mod

    return _make
