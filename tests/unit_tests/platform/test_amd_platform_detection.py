# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AMD ROCm platform detection, config file and distributed backend."""

import types

import pytest
import torch

from sglang_fl.dispatch.config import utils


@pytest.fixture
def rocm_cuda(monkeypatch):
    """Make torch look like a ROCm build with a working torch.cuda."""
    monkeypatch.setattr(torch.version, "hip", "7.2.0", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for attr in ("txda", "npu", "musa", "gcu", "corex"):
        monkeypatch.delattr(torch, attr, raising=False)
    monkeypatch.setattr(torch, "__version__", "2.10.0+rocm7.2")


def _fake_device_info(monkeypatch, vendor_name):
    """Make sglang_fl.utils.get_device_info() report `vendor_name` (None = detector unavailable)."""
    info = None if vendor_name is None else types.SimpleNamespace(vendor_name=vendor_name)
    monkeypatch.setattr("sglang_fl.utils.get_device_info", lambda: info)


def test_rocm_with_amd_vendor_is_amd(rocm_cuda, monkeypatch):
    _fake_device_info(monkeypatch, "amd")
    assert utils.get_platform_name() == "amd"


def test_rocm_with_hygon_vendor_stays_nvidia(rocm_cuda, monkeypatch):
    _fake_device_info(monkeypatch, "hygon")
    assert utils.get_platform_name() == "nvidia"


def test_nvidia_vendor_stays_nvidia(rocm_cuda, monkeypatch):
    _fake_device_info(monkeypatch, "nvidia")
    assert utils.get_platform_name() == "nvidia"


def test_detector_unavailable_falls_back_to_nvidia(rocm_cuda, monkeypatch):
    _fake_device_info(monkeypatch, None)
    assert utils.get_platform_name() == "nvidia"


def test_amd_config_file_exists():
    path = utils.get_config_path("amd")
    assert path is not None and path.name == "amd.yaml"


def test_amd_is_in_the_dist_backend_map():
    pytest.importorskip("sglang")
    from sglang_fl.platform import _DIST_BACKEND_MAP

    assert _DIST_BACKEND_MAP["amd"] == "nccl"
