"""T-Head PPU resolves its own dispatch config instead of nvidia.yaml."""

from types import SimpleNamespace

import pytest
import torch

from sglang_fl import utils as fl_utils
from sglang_fl.dispatch.config import utils


@pytest.fixture
def cuda_like(monkeypatch):
    for attr in ("txda", "npu", "musa", "gcu", "corex"):
        monkeypatch.delattr(torch, attr, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


@pytest.mark.parametrize("vendor,expected", [("thead", "thead"), ("nvidia", "nvidia")])
def test_cuda_api_platform_split_by_vendor(monkeypatch, cuda_like, vendor, expected):
    monkeypatch.setattr(
        fl_utils, "get_device_info", lambda: SimpleNamespace(vendor_name=vendor)
    )
    assert utils.get_platform_name() == expected


def test_missing_device_detector_falls_back_to_nvidia(monkeypatch, cuda_like):
    monkeypatch.setattr(fl_utils, "get_device_info", lambda: None)
    assert utils.get_platform_name() == "nvidia"


def test_thead_config_blacklists_masked_scatter():
    config = utils.load_platform_config("thead")
    assert config["prefer"] == "flagos"
    assert "masked_scatter_" in config["flagos_blacklist"]
    assert "_foreach_copy_" in config["flagos_blacklist"]
    # NVIDIA-only choices must not leak onto PPU (flagos TopK is PPU-tuned).
    assert "TopK" not in (config.get("oot_blacklist") or [])
