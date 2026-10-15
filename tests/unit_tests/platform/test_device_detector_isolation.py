# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Mocked detector results must not survive the fixture's real teardown."""

import os
from pathlib import Path
import subprocess
import sys


def test_mocked_device_info_does_not_leak_to_the_next_test(tmp_path):
    (tmp_path / "conftest.py").write_text(
        'pytest_plugins = ["tests.unit_tests.platform.conftest"]\n',
        encoding="utf-8",
    )
    (tmp_path / "test_detector_probe.py").write_text(
        """import sys
from types import SimpleNamespace

from sglang_fl import utils

MISSING = object()
ORIGINAL_FLAG_GEMS = sys.modules.get("flag_gems", MISSING)


def test_cache_fake_vendor(mock_device_detector):
    mock_device_detector("fakevendor_idem")
    cached = utils.get_device_info()
    assert cached.vendor_name == "fakevendor_idem"
    assert utils.get_device_info() is cached


def test_next_test_rediscovers_its_device(monkeypatch):
    assert sys.modules.get("flag_gems", MISSING) is ORIGINAL_FLAG_GEMS
    restored = SimpleNamespace(vendor_name="restored_vendor")
    monkeypatch.setattr(utils, "DeviceInfo", lambda: restored)
    assert utils.get_device_info().vendor_name == "restored_vendor"
    assert utils.get_device_info() is restored
""",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    root = str(Path(__file__).resolve().parents[3])
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (root, environment.get("PYTHONPATH")) if part
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "test_detector_probe.py",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--confcutdir={tmp_path}",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
