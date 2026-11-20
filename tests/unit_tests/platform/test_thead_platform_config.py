"""T-Head PPU resolves its own dispatch config instead of nvidia.yaml."""

from types import SimpleNamespace

import pytest

from sglang_fl import utils as fl_utils
from sglang_fl.dispatch.config import utils


@pytest.mark.parametrize("vendor,expected", [("thead", "thead"), ("nvidia", "nvidia")])
def test_cuda_api_platform_split_by_vendor(monkeypatch, vendor, expected):
    monkeypatch.setattr(
        fl_utils, "get_device_info", lambda: SimpleNamespace(vendor_name=vendor)
    )
    assert utils._resolve_cuda_platform() == expected


def test_missing_device_detector_falls_back_to_nvidia(monkeypatch):
    monkeypatch.setattr(fl_utils, "get_device_info", lambda: None)
    assert utils._resolve_cuda_platform() == "nvidia"


def test_thead_config_blacklists_masked_scatter():
    config = utils.load_platform_config("thead")
    assert config["prefer"] == "flagos"
    assert "masked_scatter_" in config["flagos_blacklist"]
    assert "_foreach_copy_" in config["flagos_blacklist"]
    # NVIDIA-only choices must not leak onto PPU (flagos TopK is PPU-tuned).
    assert "TopK" not in (config.get("oot_blacklist") or [])
