#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Collect the Runtime reference data from the pinned base digest.

Runs INSIDE a container of the pinned base image (build_two_wheel.sh and
verify_image.sh wrap the docker run) with the output directory bind-mounted
writable, and emits, under --out:

  runtime-0.5.18.runtime-freeze.txt     canonical name==version inventory
  runtime-0.5.18.site-files.sha256      site-packages file hashes
  runtime-0.5.18.pip-check-baseline.txt the runtime's own pip check output
  runtime-0.5.18.provenance.json        compiler slots + vendor module origins
  runtime-distributions-before.json     canonical map (dry-run plan gate +
                                        the CI job-time evidence pack)
  runtime-constraints.txt               canonical name==version resolver
                                        constraints (build input)

The files are reference data for the POST-BUILD acceptance only: they live
in mktemp/CI artifacts, are never committed, and are never COPYed into an
image. Collecting from the pinned digest replaces the previously committed
static snapshots; regenerating is deterministic for an immutable digest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

from packaging.utils import canonicalize_name

# Mirror the final image's container-start module resolution: PYTHONPATH
# (/opt/flagtree) precedes site-packages there. The BASE image has no
# image-level PYTHONPATH (our layer adds it), so insert it at the same
# position to record what container start will resolve.
if not os.environ.get("PYTHONPATH"):
    sys.path.insert(0, "/opt/flagtree")

# The documented provenance-v2 compiler slots of the operations Runtime.
SELECTED_SLOT = "/opt/flagtree"
KNOWN_SLOTS = (SELECTED_SLOT, "/opt/triton")
_VERSION_RE = re.compile(r"__version__\s*=\s*['\"]([^'\"]+)")


def installed_distributions() -> dict[str, str]:
    return {
        canonicalize_name(dist.metadata["Name"]): dist.version
        for dist in md.distributions()
        if dist.metadata.get("Name")
    }


def probe_module(module: str) -> dict:
    # find_spec resolves without executing the module: no device libraries
    # are touched, so the probe is meaningful on a GPU-less builder.
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        return {"found": False}
    return {"found": True, "origin": spec.origin}


def collect(out: Path, base_ref: str) -> None:
    packages = installed_distributions()
    with (out / "runtime-distributions-before.json").open("w") as f:
        json.dump(packages, f, indent=2, sort_keys=True)
    with (out / "runtime-constraints.txt").open("w") as f:
        for name, version in sorted(packages.items()):
            f.write(f"{name}=={version}\n")
    with (out / "runtime-0.5.18.runtime-freeze.txt").open("w") as f:
        for name, version in sorted(packages.items()):
            f.write(f"{name}=={version}\n")

    # File-level baseline for the runtime's site-packages: every file's
    # SHA256, excluding __pycache__/*.pyc (interpreters legitimately
    # (re)generate them at import time; they are not source drift).
    purelib = Path(sysconfig.get_paths()["purelib"])
    manifest = {}
    for path in sorted(purelib.rglob("*")):
        if not path.is_file() or path.suffix == ".pyc":
            continue
        if "__pycache__" in path.parts:
            continue
        manifest[str(path.relative_to(purelib))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    with (out / "runtime-0.5.18.site-files.sha256").open("w") as f:
        for relative, digest in manifest.items():
            f.write(f"{digest}  {relative}\n")

    # The runtime's own pip check findings, kept as the baseline so
    # pre-existing runtime findings are not attributed to the application.
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
    )
    with (out / "runtime-0.5.18.pip-check-baseline.txt").open("w") as f:
        f.write(completed.stdout)
        if completed.stderr.strip():
            f.write(completed.stderr)

    slots = {}
    for slot in KNOWN_SLOTS:
        init = Path(slot) / "triton" / "__init__.py"
        if not init.is_file():
            continue
        match = _VERSION_RE.search(init.read_text())
        slots[slot] = match.group(1) if match else ""
    assert SELECTED_SLOT in slots and slots[SELECTED_SLOT], (
        f"selected compiler slot missing or versionless in the base: {slots}"
    )

    provenance = {
        "base_image": base_ref,
        "compiler_slots": slots,
        "generated_by": (
            "docker/hygon/scripts/collect_runtime_reference.py against the "
            "pinned base digest; regenerate whenever the digest changes"
        ),
        "schema": (
            "compiler-slots-v2 (keys are slot ROOTS; each value is the "
            "triton __init__.__version__ under <root>/triton) + modules "
            "(vendor module origins for the provenance gate)"
        ),
        "selected_slot": SELECTED_SLOT,
        "selection": "image-level ENV PYTHONPATH=/opt/flagtree",
        "modules": {
            module: probe_module(module)
            for module in ("torch", "triton", "flag_gems", "flagcx", "flagtree")
        },
    }
    with (out / "runtime-0.5.18.provenance.json").open("w") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)

    print(f"reference data collected under {out}:")
    print(f"  distributions: {len(packages)} (canonical)")
    print(f"  site-packages files hashed: {len(manifest)}")
    print(f"  compiler slots: {slots}")
    print(f"  pip check exitcode: {completed.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--base-ref",
        default="",
        help="the pinned base digest, recorded in provenance.json",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    collect(args.out, args.base_ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
