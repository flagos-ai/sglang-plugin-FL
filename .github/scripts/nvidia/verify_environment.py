#!/usr/bin/env python3
"""Verify the NVIDIA operations-runtime SGLang 0.5.18 CI environment."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from importlib import metadata


# Operations runtime (flagos-runtime-nvidia-cuda13.3, digest-pinned in the
# containerfile) plus the SGLang closure layered on top by the image build.
# The runtime contributes torch/flagtree/flag-gems/numpy/SQLAlchemy; the
# SGLang wheel closure contributes sglang/sglang-kernel/flashinfer/triton.
EXPECTED_RUNTIME_DISTRIBUTIONS = {
    "sglang": "0.5.18",
    "torch": "2.13.0+cu130",
    "sglang-kernel": "0.4.6.post1",
    "flagtree": "0.7.0",
    "flag-gems": "5.4.0",
    "flashinfer-python": "0.6.17",
    "numpy": "1.26.4",
    "packaging": "26.3",
    "PyYAML": "6.0.1",
    "SQLAlchemy": "2.0.57",
}

# Installed into the ephemeral CI container by setup.sh, not baked into the
# image (the operations runtime must stay free of test packages).
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
        # Post-setup: triton must be 3.6.0 (matching FlagTree's version at
        # /opt/flagtree). The original 3.7.1 from the standard closure caused
        # kernel-launch crashes (SystemError in the nvidia driver launcher);
        # having no pip triton at all triggers FlagTree import side effects
        # that corrupt sys.modules during test-suite execution.
        try:
            standalone_triton = metadata.version("triton")
        except metadata.PackageNotFoundError:
            standalone_triton = None
        if standalone_triton != "3.6.0":
            raise RuntimeError(
                f"triton must be 3.6.0 (FlagTree's version); found {standalone_triton}"
            )
        # Post-setup: the closure's nvcc 13.5.37 mismatches the runtime's
        # CUDA 13.0 headers (nvidia-cuda-runtime 13.0.96 owns cu13/include)
        # and flashinfer's CCCL gate aborts JIT compiles. setup.sh realigns
        # the compiler to the runtime's 13.0 line.
        for ctk_pkg in ("nvidia-cuda-nvcc", "nvidia-nvvm", "nvidia-cuda-crt"):
            try:
                ctk_version = metadata.version(ctk_pkg)
            except metadata.PackageNotFoundError:
                raise RuntimeError(f"required distribution is missing: {ctk_pkg}")
            if ctk_version != "13.0.88":
                raise RuntimeError(
                    f"{ctk_pkg} must be 13.0.88 (runtime's CTK header line); "
                    f"found {ctk_version}"
                )
    # FlagTree's triton module must be what resolves (PYTHONPATH=/opt/flagtree
    # precedes site-packages at container start; find_spec resolves without
    # executing device libraries).
    triton_spec = importlib.util.find_spec("triton")
    if (
        triton_spec is None
        or not triton_spec.origin
        or not triton_spec.origin.startswith("/opt/flagtree")
    ):
        found = triton_spec.origin if triton_spec else None
        raise RuntimeError(
            f"triton module must resolve under /opt/flagtree, found {found}"
        )
    print(f"[cuda-env] triton-module authoritative at {triton_spec.origin}")

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
