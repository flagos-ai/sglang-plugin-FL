#!/usr/bin/env python3
"""Verify the Kunlunxin (P800) SGLang plugin environment.

Checks only:
  1) required packages are installed
  2) versions match expected values (optional, --check-versions)
  3) CUDA-compat devices are visible and enough (--require-gpu)
  4) with --require-gpu: the XPU platform marker, a real per-device tensor
     round-trip, and the FlagCX runtime that ``SGLANG_FL_DIST_BACKEND=flagcx``
     requires

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
    "flag_gems": "0.0.0+g75823ed",
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
        # Import with a statement, not ``importlib.import_module``: Kunlunxin
        # patches ``builtins.__import__`` to run its symbol rewrite, and the
        # importlib path bypasses that hook. Loading torch that way leaves the
        # CUDA-compat kernels unrewritten, so every later kernel fails with
        # "invalid device function".
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "torch.cuda.is_available() is false (expected Kunlunxin CUDA-compat)"
            )

        # Platform identity contract. ``get_platform_name`` keys Kunlunxin off
        # ``torch.xpu``: without that attribute it falls through to "nvidia",
        # ``dispatch/config/kunlunxin.yaml`` is never loaded, and the CUDA-compat
        # backend stays enabled (``deny_vendors: [cuda]`` silently dropped).
        # Fail here rather than later as an unrelated unit-test mismatch.
        if not hasattr(torch, "xpu"):
            raise RuntimeError(
                "torch exposes no 'xpu' attribute; sglang_fl would report the "
                "platform as 'nvidia' and skip dispatch/config/kunlunxin.yaml"
            )

        from sglang_fl.dispatch.config import utils as config_utils

        detected = config_utils.get_platform_name()
        if detected != "kunlunxin":
            raise RuntimeError(
                f"get_platform_name() returned {detected!r}, expected 'kunlunxin'; "
                "dispatch/config/kunlunxin.yaml would be skipped"
            )
        print(f"[kunlunxin-env] platform={detected}")

        count = torch.cuda.device_count()
        if count < min_gpus:
            raise RuntimeError(f"need >= {min_gpus} GPUs, found {count}")
        devices = ", ".join(
            f"{i}:{torch.cuda.get_device_name(i)}" for i in range(count)
        )
        print(f"[kunlunxin-env] visible-gpus={count} ({devices})")

        # Device visibility is not device health: the CUDA-compat layer can
        # enumerate a P800 that then fails inside a kernel. Run a tiny real
        # computation per device (same idea as Iluvatar's check.sh).
        for index in range(count):
            probe = torch.arange(4, device=f"cuda:{index}", dtype=torch.float32)
            total = probe.square().sum().item()
            torch.cuda.synchronize(index)
            if total != 14.0:
                raise RuntimeError(
                    f"cuda:{index} smoke failed: sum(square)={total} != 14.0"
                )
            print(f"[kunlunxin-env] smoke cuda:{index}={total}")

        # The CI config forces SGLANG_FL_DIST_BACKEND=flagcx; a missing runtime
        # only fails later inside process-group bootstrap.
        flagcx_root = os.environ.get("FLAGCX_PATH", "").strip() or "/opt/FlagCX"
        library = Path(flagcx_root) / "build" / "lib" / "libflagcx.so"
        if not library.is_file():
            raise RuntimeError(
                f"FlagCX library is missing: {library} "
                "(SGLANG_FL_DIST_BACKEND=flagcx is configured)"
            )
        try:
            import flagcx
        except Exception as exc:
            raise RuntimeError(
                f"import flagcx failed with FLAGCX_PATH={flagcx_root}: {exc}"
            ) from exc
        print(f"[kunlunxin-env] flagcx={getattr(flagcx, '__file__', '?')}")


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
