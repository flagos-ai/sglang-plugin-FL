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

"""Activate the CUDA Toolkit layout installed by NVIDIA Python wheels.

NVIDIA's CUDA 13 wheels deliberately install under ``site-packages/nvidia/cu13``.
FlashInfer 0.6.17 instead consumes the conventional Toolkit contract
(``CUDA_HOME/bin``, ``CUDA_HOME/include`` and ``CUDA_HOME/lib64/libcudart.so``).
This module creates that conventional *view* in a user-writable cache without
changing the Runtime or any installed distribution.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import tempfile
from importlib import metadata
from pathlib import Path

logger = logging.getLogger(__name__)


def _cuda_minor(value: str) -> tuple[int, int]:
    match = re.search(r"(?<!\d)(\d+)\.(\d+)(?!\d)", value)
    if match is None:
        raise RuntimeError(f"cannot determine CUDA major.minor from {value!r}")
    return int(match.group(1)), int(match.group(2))


def _nvcc_cuda_minor(nvcc: Path) -> tuple[int, int]:
    output = subprocess.check_output(
        [str(nvcc), "--version"], text=True, stderr=subprocess.STDOUT
    )
    release = re.search(r"release\s+(\d+\.\d+)", output)
    if release is None:
        raise RuntimeError(f"cannot parse CUDA release from {nvcc}: {output!r}")
    return _cuda_minor(release.group(1))


def _valid_cuda_home(path: Path) -> bool:
    return all(
        candidate.exists()
        for candidate in (
            path / "bin" / "nvcc",
            path / "include" / "cuda_runtime.h",
            path / "lib64" / "libcudart.so",
        )
    )


def _distribution_libcudart() -> Path:
    dist = metadata.distribution("nvidia-cuda-runtime")
    candidates = []
    for relative in dist.files or ():
        name = str(relative).replace("\\", "/")
        if re.search(r"/lib/libcudart\.so\.\d+$", name):
            candidates.append(Path(dist.locate_file(relative)).resolve())
    if len(candidates) != 1:
        raise RuntimeError(
            "expected exactly one versioned libcudart from nvidia-cuda-runtime, "
            f"found {[str(path) for path in candidates]}"
        )
    return candidates[0]


def _ensure_symlink(link: Path, target: Path) -> None:
    """Create *link* once, tolerating concurrent creators of the same link."""

    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except FileExistsError:
        # Another worker may be activating the same immutable toolkit view.
        # Accept only the exact link it was about to create.
        if link.is_symlink() and link.resolve() == target.resolve():
            return
        raise RuntimeError(
            f"refusing to replace existing CUDA activation path: {link}"
        ) from None


def _cache_root() -> Path:
    configured = os.environ.get("SGLANG_FL_CUDA_VIEW_ROOT")
    if configured:
        return Path(configured)

    # Containers often have a read-only or unset HOME.  Prefer the explicitly
    # configured XDG cache, otherwise use the process' writable temporary root.
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache) / "sglang-fl" / "cuda"
    return Path(tempfile.gettempdir()) / f"sglang-fl-{os.getuid()}" / "cuda"


def _create_compatibility_view(
    *, nvcc: Path, include_dir: Path, libcudart: Path
) -> Path:
    identity = hashlib.sha256(
        (
            "v2\0"
            f"{nvcc.resolve()}\0{include_dir.resolve()}\0{libcudart.resolve()}"
        ).encode()
    ).hexdigest()[:16]
    cache_root = _cache_root()
    view = cache_root / identity
    lib64 = view / "lib64"
    try:
        lib64.mkdir(parents=True, exist_ok=True)
        _ensure_symlink(view / "bin", nvcc.parent)
        _ensure_symlink(view / "include", include_dir)
        # nvcc links libcudadevrt.a and libcudart_static.a implicitly in
        # addition to FlashInfer's explicit -lcudart. Expose the complete
        # wheel-owned runtime-lib directory, not only the dynamic library.
        for library in libcudart.parent.iterdir():
            if library.is_file():
                _ensure_symlink(lib64 / library.name, library)
        _ensure_symlink(lib64 / "libcudart.so", libcudart)
    except OSError as exc:
        raise RuntimeError(
            f"cannot create NVIDIA CUDA compatibility view under {cache_root}: {exc}"
        ) from exc
    return view


def _prepend_env(name: str, *paths: Path) -> None:
    current = [item for item in os.environ.get(name, "").split(os.pathsep) if item]
    wanted = [str(path) for path in paths]
    os.environ[name] = os.pathsep.join(wanted + [x for x in current if x not in wanted])


def _is_nvidia_vendor() -> bool:
    # torch.version.cuda is not a hardware identity: T-Head PPU intentionally
    # exposes the CUDA API and reports CUDA 13.  FlagGems' DeviceDetector is the
    # shared source of truth and distinguishes that Runtime as vendor=thead.
    from sglang_fl.utils import get_device_info

    info = get_device_info()
    return info is not None and info.vendor_name == "nvidia"


def activate_nvidia_runtime(torch_module) -> Path | None:
    """Activate the pip CUDA layout for NVIDIA, returning its compatibility view.

    A complete user-provided ``CUDA_HOME``/``CUDA_PATH`` wins.  Non-NVIDIA
    backends are a no-op.  NVIDIA environments fail early when the compiler,
    headers and runtime library disagree with the CUDA line used by PyTorch.
    """

    if not _is_nvidia_vendor():
        return None

    torch_cuda = getattr(getattr(torch_module, "version", None), "cuda", None)
    torch_hip = getattr(getattr(torch_module, "version", None), "hip", None)
    if torch_cuda is None or torch_hip is not None or hasattr(torch_module, "corex"):
        return None

    expected = _cuda_minor(torch_cuda)
    configured = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if configured and _valid_cuda_home(Path(configured)):
        configured_path = Path(configured)
        actual = _nvcc_cuda_minor(configured_path / "bin" / "nvcc")
        if actual != expected:
            raise RuntimeError(
                "CUDA compiler does not match the PyTorch CUDA contract: "
                f"torch={expected[0]}.{expected[1]}, "
                f"nvcc={actual[0]}.{actual[1]} ({configured_path / 'bin' / 'nvcc'})"
            )
        return configured_path

    try:
        from cuda.pathfinder import (
            find_nvidia_binary_utility,
            find_nvidia_header_directory,
        )
    except ImportError as exc:
        raise RuntimeError(
            "NVIDIA activation requires cuda-pathfinder from the CUDA toolkit"
        ) from exc

    nvcc_value = find_nvidia_binary_utility("nvcc")
    include_value = find_nvidia_header_directory("cudart")
    if nvcc_value is None or include_value is None:
        raise RuntimeError(
            "NVIDIA CUDA wheel layout is incomplete: "
            f"nvcc={nvcc_value!r}, cudart_headers={include_value!r}"
        )
    nvcc = Path(nvcc_value)
    include_dir = Path(include_value)
    libcudart = _distribution_libcudart()

    actual = _nvcc_cuda_minor(nvcc)
    if actual != expected:
        raise RuntimeError(
            "CUDA compiler does not match the PyTorch CUDA contract: "
            f"torch={expected[0]}.{expected[1]}, nvcc={actual[0]}.{actual[1]} "
            f"({nvcc})"
        )

    view = _create_compatibility_view(
        nvcc=nvcc, include_dir=include_dir, libcudart=libcudart
    )
    os.environ["CUDA_HOME"] = str(view)
    os.environ["CUDA_PATH"] = str(view)
    _prepend_env("PATH", view / "bin")
    _prepend_env("LD_LIBRARY_PATH", view / "lib64", libcudart.parent)
    logger.info(
        "Activated NVIDIA CUDA wheel layout: torch=%s, nvcc=%s, view=%s",
        torch_cuda,
        nvcc,
        view,
    )
    return view
