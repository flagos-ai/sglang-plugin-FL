#!/usr/bin/env python3
"""Verify the pinned NVIDIA SGLang 0.5.18 CI environment."""

from __future__ import annotations

import argparse
import importlib
import sys
from importlib import metadata


EXPECTED_RUNTIME_DISTRIBUTIONS = {
    "sglang": "0.5.18",
    "torch": "2.13.0+cu130",
    "sglang-kernel": "0.4.6.post1",
    "flagtree": "0.6.2a1",
    "flag-gems": "5.3.6.dev0+g8ea5925",
    "flashinfer-python": "0.6.17",
    "numpy": "2.3.5",
    "packaging": "26.3",
    "PyYAML": "6.0.1",
    "SQLAlchemy": "2.0.48",
}

EXPECTED_CI_DISTRIBUTIONS = {
    "pytest": "9.1.1",
    "pytest-timeout": "2.4.0",
}


def _require_distribution(name: str, expected: str) -> None:
    try:
        actual = metadata.version(name)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required distribution is missing: {name}") from exc
    if actual != expected:
        raise RuntimeError(f"{name} must be {expected}, found {actual}")
    print(f"[cuda-env] {name}={actual}")


def verify(
    *, require_gpu: bool = False, require_ci: bool = False, min_gpus: int = 1
) -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"Python must be 3.12, found {sys.version_info.major}.{sys.version_info.minor}"
        )
    print(f"[cuda-env] python={sys.version.split()[0]}")

    for name, expected in EXPECTED_RUNTIME_DISTRIBUTIONS.items():
        _require_distribution(name, expected)
    if require_ci:
        for name, expected in EXPECTED_CI_DISTRIBUTIONS.items():
            _require_distribution(name, expected)

    try:
        standalone_triton = metadata.version("triton")
    except metadata.PackageNotFoundError:
        standalone_triton = None
    if standalone_triton is not None:
        raise RuntimeError(
            "the standalone triton distribution must be uninstalled; "
            f"found triton={standalone_triton}"
        )

    triton = importlib.import_module("triton")
    triton_version = getattr(triton, "__version__", "unknown")
    if triton_version != "3.6.0":
        raise RuntimeError(
            f"FlagTree must provide triton 3.6.0, found {triton_version}"
        )
    print(f"[cuda-env] triton-module={triton_version} (provided by FlagTree)")

    torch = importlib.import_module("torch")
    if torch.version.cuda != "13.0":
        raise RuntimeError(f"PyTorch must use CUDA 13.0, found {torch.version.cuda}")
    print(f"[cuda-env] torch-cuda={torch.version.cuda}")

    for module_name in ("sglang", "flashinfer"):
        importlib.import_module(module_name)
        print(f"[cuda-env] imported {module_name}")

    kernel_files = metadata.files("sglang-kernel") or []
    normalized_kernel_files = {str(path).replace("\\", "/") for path in kernel_files}
    if not any(
        path.startswith("sgl_kernel/") and "/common_ops." in path
        for path in normalized_kernel_files
    ):
        raise RuntimeError("sglang-kernel wheel contains no common_ops library")
    print("[cuda-env] sglang-kernel wheel contains common_ops")

    flag_gems_files = metadata.files("flag-gems") or []
    normalized_files = {str(path).replace("\\", "/") for path in flag_gems_files}
    dsa_package = "flag_gems/fused/DSA/__init__.py"
    if dsa_package not in normalized_files:
        raise RuntimeError(f"FlagGems wheel is missing {dsa_package}")
    print(f"[cuda-env] FlagGems wheel contains {dsa_package}")

    if require_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")
        device_count = torch.cuda.device_count()
        if device_count < min_gpus:
            raise RuntimeError(
                f"at least {min_gpus} CUDA devices are required, found {device_count}"
            )
        devices = ", ".join(
            f"{index}:{torch.cuda.get_device_name(index)}"
            for index in range(device_count)
        )
        print(f"[cuda-env] visible-gpus={device_count} ({devices})")
        importlib.import_module("sgl_kernel")
        importlib.import_module("flag_gems")
        importlib.import_module("flag_gems.fused.DSA")
        print(
            "[cuda-env] sgl_kernel, FlagGems, and flag_gems.fused.DSA imports succeeded"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="also require at least one CUDA device visible to PyTorch",
    )
    parser.add_argument(
        "--require-ci",
        action="store_true",
        help="also require the pinned test-runner distributions",
    )
    parser.add_argument(
        "--min-gpus",
        type=int,
        default=1,
        help="minimum visible GPU count when --require-gpu is enabled",
    )
    args = parser.parse_args()
    if args.min_gpus < 1:
        parser.error("--min-gpus must be at least 1")
    verify(
        require_gpu=args.require_gpu,
        require_ci=args.require_ci,
        min_gpus=args.min_gpus,
    )
    print("[cuda-env] environment verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
