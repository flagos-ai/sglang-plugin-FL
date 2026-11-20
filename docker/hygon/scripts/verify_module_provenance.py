#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Post-build gate: vendor module provenance (external phase).

Runs INSIDE the finished image (docker/hygon/scripts/verify_image.sh wraps
the docker run) and compares the live module origins of the vendor stack
against the reference provenance collected dynamically from the digest the
build resolved once from the Runtime's moving tag. The image-level
PYTHONPATH=/opt/flagtree makes the runtime's
FlagTree tree the authoritative resolution in every entrypoint (this
process included), and the standalone triton wheel never enters
resolution, so every vendor module must resolve to the exact same origin
before and after the closure install.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

VENDOR_MODULES = ("torch", "triton", "flag_gems", "flagcx", "flagtree")

# The active Triton slot (image-level PYTHONPATH); the reference was
# collected with the production /opt/flagtree selection, so under the
# experiment's vendor slot only triton's expected origin changes - every
# other module must match the reference exactly.
EXPECTED_TRITON_ROOT = os.environ.get("SGLANG_FL_EXPECTED_TRITON_ROOT", "/opt/flagtree")

# Mirror the container-start module resolution: PYTHONPATH (/opt/flagtree)
# precedes site-packages there, while plain build steps do not inherit
# that ENV. Same position, same precedence as at container start.
if not os.environ.get("PYTHONPATH"):
    sys.path.insert(0, "/opt/flagtree")


def probe_module(module: str) -> dict:
    # find_spec resolves without executing the module: no device libraries
    # are touched, so the probe is meaningful on a GPU-less builder.
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        return {"found": False}
    return {"found": True, "origin": spec.origin}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help=(
            "provenance.json collected from the build's resolved base digest "
            "(carries the recorded 'modules' origins)"
        ),
    )
    args = parser.parse_args()

    recorded = json.loads(args.reference.read_text())
    before = recorded.get("modules")
    if not before:
        print(
            f"FATAL: {args.reference} carries no vendor module origins",
            file=sys.stderr,
        )
        return 1

    after = {module: probe_module(module) for module in VENDOR_MODULES}

    def slot_expectation(module: str, probe: dict) -> bool | None:
        """Slot-geometry modules follow the active slot's contracted state.

        triton resolves from the expected slot root; the flagtree module
        (only exposed through /opt/flagtree) is contracted absent under
        the vendor slot. Everything else must match the reference exactly.
        """
        if module == "triton":
            return (
                (probe or {}).get("origin", "").startswith(EXPECTED_TRITON_ROOT + "/")
            )
        if module == "flagtree" and EXPECTED_TRITON_ROOT != "/opt/flagtree":
            return probe == {"found": False}
        return None

    changed = {
        module: (before.get(module), after.get(module))
        for module in sorted(set(before) | set(after))
        if before.get(module) != after.get(module)
        and slot_expectation(module, after.get(module)) is not True
    }
    if changed:
        print("FATAL: vendor module provenance changed across the install:")
        for module, (was, now) in changed.items():
            print(f"  ! {module}: {was} -> {now}")
        return 1
    print(
        "module provenance unchanged under container-start resolution order "
        "(triton resolves via /opt/flagtree before site-packages):"
    )
    print(json.dumps(after, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
