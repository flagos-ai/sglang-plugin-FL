# Tests for PlatformFL.init_backend vendor module auto-import.

import importlib
import logging

import pytest

from sglang_fl.platform import PlatformFL


def _make_platform_stub(vendor_name, device_type="test_device"):
    """Build a PlatformFL stub that bypasses ``__init__``.

    The real ``__init__`` calls FlagGems' ``DeviceDetector`` which needs actual
    hardware; ``init_backend`` only reads ``_vendor_name`` and ``_device_type``,
    so a bare attribute stub is enough for isolated testing.
    """
    p = PlatformFL.__new__(PlatformFL)
    p._vendor_name = vendor_name
    p._device_type = device_type
    return p


class TestInitBackendVendorAutoImport:
    def test_loaded_when_register_platform_exists(
        self, caplog, inject_vendor_module
    ):
        inject_vendor_module("fakevendor", "register_platform")
        p = _make_platform_stub("fakevendor")
        with caplog.at_level(logging.INFO, logger="sglang_fl.platform"):
            p.init_backend()
        assert "vendor_module=loaded" in caplog.text
        assert "vendor=fakevendor" in caplog.text

    def test_absent_when_register_platform_missing(self, caplog):
        p = _make_platform_stub("nonexistent_vendor_xyz_for_test")
        with caplog.at_level(logging.INFO, logger="sglang_fl.platform"):
            p.init_backend()
        assert "vendor_module=absent" in caplog.text

    def test_non_import_error_propagates(self, monkeypatch):
        """Only ImportError is treated as 'absent'. Any other exception from
        the vendor module's own import-time code must bubble up so real bugs
        are visible instead of silently logged as 'absent'.
        """
        real_import = importlib.import_module

        def _failing_import(name, *args, **kwargs):
            if name.endswith(".fakevendor.register_platform"):
                raise RuntimeError("vendor module raised on import")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", _failing_import)

        p = _make_platform_stub("fakevendor")
        with pytest.raises(RuntimeError, match="raised on import"):
            p.init_backend()

    def test_idempotent_repeated_calls(self, caplog, inject_vendor_module):
        """Repeated init_backend calls hit importlib's module cache — the
        vendor module's top-level code only executes once, subsequent calls
        log 'loaded' but otherwise no-op. Matters for multi-spawn scenarios
        where each worker re-enters init_backend.
        """
        inject_vendor_module("fakevendor_idem", "register_platform")
        p = _make_platform_stub("fakevendor_idem")
        with caplog.at_level(logging.INFO, logger="sglang_fl.platform"):
            p.init_backend()
            p.init_backend()
            p.init_backend()
        assert caplog.text.count("vendor_module=loaded") == 3
