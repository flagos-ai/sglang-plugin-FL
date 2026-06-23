# Tests for _apply_vendor_patches in sglang_fl/__init__.py.

import importlib
import logging

import pytest

from sglang_fl import _apply_vendor_patches


class TestApplyVendorPatches:
    def test_loaded_when_patch_exists(
        self, caplog, mock_device_detector, inject_vendor_module
    ):
        mock_device_detector("fakevendor")
        inject_vendor_module("fakevendor", "patch")
        with caplog.at_level(logging.INFO, logger="sglang_fl"):
            _apply_vendor_patches()
        assert "vendor patch loaded" in caplog.text
        assert "fakevendor.patch" in caplog.text

    def test_absent_when_patch_missing(self, caplog, mock_device_detector):
        mock_device_detector("nonexistent_vendor_xyz_for_test")
        with caplog.at_level(logging.INFO, logger="sglang_fl"):
            _apply_vendor_patches()
        assert "vendor patch absent" in caplog.text

    def test_detector_failure_skips_without_raising(
        self, caplog, mock_device_detector
    ):
        """DeviceDetector raising must not crash load_plugin — vendor patch is
        optional. Expected behaviour: warning log + early return, no attempt
        to import patch.py."""
        mock_device_detector(raise_exc=RuntimeError("no hardware found"))
        with caplog.at_level(logging.WARNING, logger="sglang_fl"):
            _apply_vendor_patches()
        assert "vendor patch skipped" in caplog.text
        assert "no hardware found" in caplog.text

    def test_non_import_error_propagates(
        self, mock_device_detector, monkeypatch
    ):
        """Only ImportError is treated as 'absent'. Any other exception from
        the vendor patch.py's own import-time code must bubble up."""
        mock_device_detector("fakevendor")

        real_import = importlib.import_module

        def _failing_import(name, *args, **kwargs):
            if name.endswith(".fakevendor.patch"):
                raise RuntimeError("patch module crashed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", _failing_import)

        with pytest.raises(RuntimeError, match="patch module crashed"):
            _apply_vendor_patches()

    def test_idempotent_repeated_calls(
        self, caplog, mock_device_detector, inject_vendor_module
    ):
        """Repeated calls hit importlib's cache: patch.py's top-level code
        runs once, subsequent calls log 'loaded' but produce no extra side
        effects. Matters because each worker process re-enters load_plugin.
        """
        mock_device_detector("fakevendor_idem")
        inject_vendor_module("fakevendor_idem", "patch")

        with caplog.at_level(logging.INFO, logger="sglang_fl"):
            _apply_vendor_patches()
            _apply_vendor_patches()
            _apply_vendor_patches()
        assert caplog.text.count("vendor patch loaded") == 3
