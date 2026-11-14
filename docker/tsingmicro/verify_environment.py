#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the TsingMicro (TXDA) SGLang 0.5.18 empty runtime.

--metadata-only  : package metadata and module locations only (no device, no
                   torch_txda import) -- safe inside ``docker build``.
--require-device : additionally import torch_txda and require --min-cards
                   visible TXDA devices (CI setup / check steps).

Exits non-zero with one line per problem.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

SGLANG_VERSION = "0.5.18"

# CUDA-only wheels of the regular SGLang install; must not be present.
CUDA_ONLY = {
    "sglang-kernel",
    "flashinfer-python",
    "flashinfer-cubin",
    "flash-attn-4",
    "vllm",
    "kernels",
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "nvidia-cudnn-frontend",
    "nvidia-ml-py",
    "nvgpu",
    "cuda-python",
    "cuda-bindings",
    "cuda-core",
    "cuda-tile",
    "quack-kernels",
    "torch-memory-saver",
}

# Second SGLang environments that must not exist (single-environment image).
FORBIDDEN_PATHS = ["/sgl-workspace-0.5.18/venv", "/sgl-workspace/sglang"]

# Known, accepted `pip check` findings of the TXDA vendor stack.
#   - FlagTree provides the ``triton`` module without a ``triton`` distribution.
#   - FlagGems pins packaging==26.0, torch pins setuptools<82; both work.
#   - opencv / sentence-transformers are unrelated base-image extras.
#   - compressed-tensors 0.15.0 (the pin of all 0.5.18 vendor images, installed
#     --no-deps) declares torch<2.11; TXDA torch is 2.11.0+cpu. Accepted: the
#     validated 0.5.18 runs used exactly this pair.
#   - the base image's ninja wheel reports "not supported on this platform".
ALLOWED_PIP_CHECK = [
    r"^(xgrammar|compressed-tensors) .* requires triton",
    r"^compressed-tensors 0\.15\.0 has requirement torch<2\.11",
    r"^ninja .* is not supported on this platform",
    r"^flag-gems .* has requirement packaging==",
    r"^torch .* has requirement setuptools<",
    r"^opencv-python",
    r"^sentence-transformers",
    r"^pygobject .* requires pycairo",
]


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def check_metadata(plugin_root: Path) -> list[str]:
    problems: list[str] = []

    # Only installed distributions: CI puts the checkout itself on PYTHONPATH,
    # which would also expose its in-tree *.egg-info as a second "sglang_fl".
    root = plugin_root.resolve()
    dists = {}
    for d in md.distributions():
        name = d.metadata["Name"]
        loc = getattr(d, "_path", None)
        if loc is not None and Path(loc).resolve().is_relative_to(root):
            continue
        if name:
            dists.setdefault(_norm(name), []).append(d.version)

    sglang = dists.get("sglang", [])
    if sglang != [SGLANG_VERSION]:
        problems.append(
            f"sglang distributions {sglang}, expected exactly [{SGLANG_VERSION}]"
        )
    plugin = dists.get("sglang-fl", [])
    if len(plugin) != 1:
        problems.append(f"sglang_fl distributions {plugin}, expected exactly one")

    leaked = sorted(CUDA_ONLY & set(dists))
    if leaked:
        problems.append(f"CUDA-only packages installed: {leaked}")

    for name in ("torch", "torch-txda", "flagtree", "flag-gems"):
        if name not in dists:
            problems.append(f"vendor package {name} missing")
    if "triton" in dists:
        problems.append(
            "standalone triton distribution installed (would shadow FlagTree)"
        )

    spec = importlib.util.find_spec("triton")
    if spec is None or not spec.origin:
        problems.append("triton module not importable")
    elif not (Path(spec.origin).parent / "backends" / "tsingmicro").is_dir():
        problems.append(
            f"triton module at {spec.origin} has no tsingmicro backend (not FlagTree)"
        )

    spec = importlib.util.find_spec("sglang_fl")
    if spec is None or not spec.origin:
        problems.append("sglang_fl not importable")
    elif Path(spec.origin).resolve().parent != (plugin_root / "sglang_fl").resolve():
        problems.append(
            f"sglang_fl resolves to {spec.origin}, expected {plugin_root}/sglang_fl"
        )

    for p in FORBIDDEN_PATHS:
        if os.path.exists(p):
            problems.append(f"second Python environment present: {p}")

    out = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
    ).stdout
    for line in filter(None, (raw.strip() for raw in out.splitlines())):
        if line.startswith("No broken requirements"):
            continue
        if not any(re.search(p, line) for p in ALLOWED_PIP_CHECK):
            problems.append(f"pip check: {line}")
    return problems


def check_device(min_cards: int) -> list[str]:
    try:
        import torch
        import torch_txda  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return [f"cannot import torch_txda: {exc}"]
    if not (hasattr(torch, "txda") and torch.txda.is_available()):
        return ["torch.txda is not available (device nodes / driver?)"]
    count = torch.txda.device_count()
    if count < min_cards:
        return [f"{count} visible TXDA devices, need {min_cards}"]
    print(f"torch_txda: {count} visible TXDA devices")
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plugin-root", type=Path, default=Path.cwd())
    ap.add_argument("--metadata-only", action="store_true")
    ap.add_argument("--require-device", action="store_true")
    ap.add_argument("--min-cards", type=int, default=8)
    args = ap.parse_args()

    problems = check_metadata(args.plugin_root)
    if args.require_device and not args.metadata_only:
        problems += check_device(args.min_cards)

    for name in (
        "sglang",
        "sglang_fl",
        "torch",
        "torch_txda",
        "flagtree",
        "flag_gems",
        "transformers",
        "xgrammar",
    ):
        try:
            print(f"{name}: {md.version(name)}")
        except md.PackageNotFoundError:
            print(f"{name}: <missing>")
    if problems:
        for p in problems:
            print(f"::error::{p}", file=sys.stderr)
        return 1
    print("TsingMicro SGLang 0.5.18 environment OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
