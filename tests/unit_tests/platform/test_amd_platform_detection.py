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

import sys
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


def _fake_detector(monkeypatch, vendor_name, module="device"):
    """Fake FlagGems' DeviceDetector at runtime.backend.<module>."""
    mod = types.ModuleType(f"flag_gems.runtime.backend.{module}")
    mod.DeviceDetector = lambda: types.SimpleNamespace(vendor_name=vendor_name)
    for name in ("flag_gems", "flag_gems.runtime", "flag_gems.runtime.backend"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, f"flag_gems.runtime.backend.{module}", mod)
    if module != "device":  # newer FlagGems has no runtime.backend.device
        monkeypatch.setitem(sys.modules, "flag_gems.runtime.backend.device", None)


def test_rocm_with_amd_vendor_is_amd(rocm_cuda, monkeypatch):
    _fake_detector(monkeypatch, "amd")
    assert utils.get_platform_name() == "amd"


def test_newer_flaggems_module_path_is_amd(rocm_cuda, monkeypatch):
    _fake_detector(monkeypatch, "amd", module="device_finder")
    assert utils.get_platform_name() == "amd"


def test_rocm_with_hygon_vendor_stays_nvidia(rocm_cuda, monkeypatch):
    _fake_detector(monkeypatch, "hygon")
    assert utils.get_platform_name() == "nvidia"


def test_cuda_torch_stays_nvidia(rocm_cuda, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert utils.get_platform_name() == "nvidia"


def test_detector_failure_falls_back_to_nvidia(rocm_cuda, monkeypatch):
    monkeypatch.setitem(sys.modules, "flag_gems", None)
    assert utils.get_platform_name() == "nvidia"


def test_amd_config_file_exists():
    path = utils.get_config_path("amd")
    assert path is not None and path.name == "amd.yaml"


def test_amd_is_in_the_dist_backend_map():
    pytest.importorskip("sglang")
    from sglang_fl.platform import _DIST_BACKEND_MAP

    assert _DIST_BACKEND_MAP["amd"] == "nccl"
