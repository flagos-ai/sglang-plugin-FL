#!/usr/bin/env python3
"""Verify the Kunlunxin (P800) SGLang plugin environment.

Checks only:
  1) required packages are installed
  2) versions match expected values (optional, --check-versions)
  3) CUDA-compat devices are visible and enough (--require-gpu)

Versions reference the validated container limy-sglang-upgrade.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from importlib import metadata
from pathlib import Path

# pip / importlib.metadata names (flag_gems, not flag-gems).
REQUIRED_DISTRIBUTIONS = (
    "sglang",
    "torch",
    "flagtree",
    "flag_gems",
)

# Optional exact/release pins used only with --check-versions.
# torch/sglang compare the release segment before '+'; others compare full string.
EXPECTED_VERSIONS = {
    "sglang": "0.5.18",
    "torch": "2.9.0",
    "flagtree": "0.7.1a1+xpu3.6",
    "flag_gems": "0.0+g7fcdbed07",
}


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required package is missing: {name}") from exc


def _version_ok(name: str, actual: str, expected: str) -> bool:
    if name in ("sglang", "torch"):
        return actual.split("+", 1)[0] == expected
    return actual == expected


def verify(
    *,
    check_versions: bool = False,
    require_gpu: bool = False,
    min_gpus: int = 1,
) -> None:
    if min_gpus < 1:
        raise RuntimeError("--min-gpus must be at least 1")

    print(f"[kunlunxin-env] python={sys.version.split()[0]}")

    for name in REQUIRED_DISTRIBUTIONS:
        actual = _version(name)
        if check_versions:
            expected = EXPECTED_VERSIONS[name]
            if not _version_ok(name, actual, expected):
                raise RuntimeError(f"{name} must be {expected}, found {actual}")
        print(f"[kunlunxin-env] {name}={actual}")

    # Paths (no file hashing): plugin module + optional FlagCX library.
    spec = importlib.util.find_spec("sglang_fl")
    if spec is None or spec.origin is None:
        raise RuntimeError("sglang_fl is not importable")
    print(f"[kunlunxin-env] sglang_fl path={Path(spec.origin).resolve()}")

    if require_gpu:
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "torch.cuda.is_available() is false (expected Kunlunxin CUDA-compat)"
            )
        count = torch.cuda.device_count()
        if count < min_gpus:
            raise RuntimeError(f"need >= {min_gpus} GPUs, found {count}")
        devices = ", ".join(
            f"{i}:{torch.cuda.get_device_name(i)}" for i in range(count)
        )
        print(f"[kunlunxin-env] visible-gpus={count} ({devices})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-versions",
        action="store_true",
        help="also require package versions to match EXPECTED_VERSIONS",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="require torch.cuda devices (CUDA-compat XPU)",
    )
    parser.add_argument(
        "--min-gpus",
        type=int,
        default=1,
        help="minimum visible GPU count when --require-gpu is set",
    )
    args = parser.parse_args(argv)
    try:
        verify(
            check_versions=args.check_versions,
            require_gpu=args.require_gpu,
            min_gpus=args.min_gpus,
        )
    except RuntimeError as exc:
        print(f"[kunlunxin-env] FAILED: {exc}", file=sys.stderr)
        return 1
    print("[kunlunxin-env] environment verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
