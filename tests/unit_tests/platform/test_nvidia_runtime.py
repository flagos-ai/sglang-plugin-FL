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

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import tempfile

import pytest

from sglang_fl import nvidia_runtime
from sglang_fl import utils


@pytest.mark.parametrize("vendor", [None, "thead", "hygon", "iluvatar", "ascend"])
def test_nvidia_vendor_gate_rejects_other_platforms(monkeypatch, vendor):
    info = None if vendor is None else SimpleNamespace(vendor_name=vendor)
    monkeypatch.setattr(utils, "get_device_info", lambda: info)
    assert not nvidia_runtime._is_nvidia_vendor()


def test_nvidia_vendor_gate_accepts_nvidia(monkeypatch):
    monkeypatch.setattr(
        utils, "get_device_info", lambda: SimpleNamespace(vendor_name="nvidia")
    )
    assert nvidia_runtime._is_nvidia_vendor()


@pytest.mark.parametrize(
    ("vendor", "torch_module"),
    [
        ("thead", SimpleNamespace(version=SimpleNamespace(cuda="13.0", hip=None))),
        ("hygon", SimpleNamespace(version=SimpleNamespace(cuda=None, hip="6.3"))),
        (
            "iluvatar",
            SimpleNamespace(
                version=SimpleNamespace(cuda="12.4", hip=None), corex=object()
            ),
        ),
    ],
)
def test_non_nvidia_activation_is_a_noop(monkeypatch, vendor, torch_module):
    monkeypatch.setattr(
        utils, "get_device_info", lambda: SimpleNamespace(vendor_name=vendor)
    )
    before = dict(nvidia_runtime.os.environ)
    assert nvidia_runtime.activate_nvidia_runtime(torch_module) is None
    assert dict(nvidia_runtime.os.environ) == before


def _fake_nvcc(path: Path, release: str = "13.0") -> Path:
    path.parent.mkdir(parents=True)
    path.write_text(
        f"#!/bin/sh\necho 'Cuda compilation tools, release {release}, V0.0.0'\n"
    )
    path.chmod(0o755)
    return path


def _fake_toolkit(root: Path, release: str = "13.0") -> tuple[Path, Path, Path]:
    nvcc = _fake_nvcc(root / "wheel" / "bin" / "nvcc", release)
    include = root / "wheel" / "include"
    include.mkdir(parents=True)
    (include / "cuda_runtime.h").touch()
    lib = root / "wheel" / "lib" / "libcudart.so.13"
    lib.parent.mkdir(parents=True)
    lib.touch()
    (lib.parent / "libcudadevrt.a").touch()
    (lib.parent / "libcudart_static.a").touch()
    return nvcc, include, lib


def test_compatibility_view_is_concurrently_idempotent(tmp_path, monkeypatch):
    nvcc, include, libcudart = _fake_toolkit(tmp_path)
    monkeypatch.setenv("SGLANG_FL_CUDA_VIEW_ROOT", str(tmp_path / "views"))

    def create(_):
        return nvidia_runtime._create_compatibility_view(
            nvcc=nvcc, include_dir=include, libcudart=libcudart
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        views = list(pool.map(create, range(128)))

    assert len(set(views)) == 1
    view = views[0]
    assert (view / "bin").resolve() == nvcc.parent.resolve()
    assert (view / "include").resolve() == include.resolve()
    assert (view / "lib64" / "libcudart.so").resolve() == libcudart.resolve()
    for name in ("libcudadevrt.a", "libcudart_static.a"):
        assert (view / "lib64" / name).resolve() == (libcudart.parent / name).resolve()


def test_default_cache_does_not_depend_on_home(tmp_path, monkeypatch):
    monkeypatch.delenv("SGLANG_FL_CUDA_VIEW_ROOT", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "read-only-home"))
    assert str(nvidia_runtime._cache_root()).startswith(tempfile.gettempdir())


def test_complete_configured_cuda_home_is_preserved(tmp_path, monkeypatch):
    nvcc, include, libcudart = _fake_toolkit(tmp_path)
    home = tmp_path / "configured"
    (home / "bin").mkdir(parents=True)
    (home / "include").mkdir()
    (home / "lib64").mkdir()
    (home / "bin" / "nvcc").symlink_to(nvcc)
    (home / "include" / "cuda_runtime.h").symlink_to(
        include / "cuda_runtime.h"
    )
    (home / "lib64" / "libcudart.so").symlink_to(libcudart)
    monkeypatch.setenv("CUDA_HOME", str(home))
    monkeypatch.setattr(nvidia_runtime, "_is_nvidia_vendor", lambda: True)
    torch_module = SimpleNamespace(version=SimpleNamespace(cuda="13.0", hip=None))

    assert nvidia_runtime.activate_nvidia_runtime(torch_module) == home
    assert Path.home() != home


def test_configured_cuda_home_version_mismatch_fails(tmp_path, monkeypatch):
    nvcc, include, libcudart = _fake_toolkit(tmp_path, "13.5")
    home = tmp_path / "configured"
    (home / "bin").mkdir(parents=True)
    (home / "include").mkdir()
    (home / "lib64").mkdir()
    (home / "bin" / "nvcc").symlink_to(nvcc)
    (home / "include" / "cuda_runtime.h").symlink_to(
        include / "cuda_runtime.h"
    )
    (home / "lib64" / "libcudart.so").symlink_to(libcudart)
    monkeypatch.setenv("CUDA_HOME", str(home))
    monkeypatch.setattr(nvidia_runtime, "_is_nvidia_vendor", lambda: True)
    torch_module = SimpleNamespace(version=SimpleNamespace(cuda="13.0", hip=None))

    with pytest.raises(RuntimeError, match=r"torch=13\.0, nvcc=13\.5"):
        nvidia_runtime.activate_nvidia_runtime(torch_module)


def test_activation_creates_flashinfer_layout(tmp_path, monkeypatch):
    # Only meaningful where the CUDA python wheel exists (NVIDIA-style
    # environments); skip cleanly on other vendors' images so the shared
    # unit suite stays green everywhere.
    cuda_pathfinder = pytest.importorskip(
        "cuda.pathfinder", reason="the cuda python wheel is not installed"
    )
    nvcc, include, libcudart = _fake_toolkit(tmp_path)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/lib")
    monkeypatch.setenv("SGLANG_FL_CUDA_VIEW_ROOT", str(tmp_path / "views"))
    monkeypatch.setattr(nvidia_runtime, "_is_nvidia_vendor", lambda: True)
    monkeypatch.setattr(nvidia_runtime, "_distribution_libcudart", lambda: libcudart)

    monkeypatch.setattr(
        cuda_pathfinder, "find_nvidia_binary_utility", lambda name: str(nvcc)
    )
    monkeypatch.setattr(
        cuda_pathfinder, "find_nvidia_header_directory", lambda name: str(include)
    )
    torch_module = SimpleNamespace(version=SimpleNamespace(cuda="13.0", hip=None))

    view = nvidia_runtime.activate_nvidia_runtime(torch_module)
    assert view is not None
    assert Path(nvidia_runtime.os.environ["CUDA_HOME"]) == view
    assert Path(nvidia_runtime.os.environ["CUDA_PATH"]) == view
    assert nvidia_runtime.os.environ["PATH"].split(":").count(str(view / "bin")) == 1
    assert (
        nvidia_runtime.os.environ["LD_LIBRARY_PATH"]
        .split(":")
        .count(str(view / "lib64"))
        == 1
    )

    # Repeated activation is stable and does not duplicate environment paths.
    assert nvidia_runtime.activate_nvidia_runtime(torch_module) == view
    assert nvidia_runtime.os.environ["PATH"].split(":").count(str(view / "bin")) == 1
