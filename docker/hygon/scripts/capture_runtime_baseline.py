#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Capture the pre-install runtime state into the evidence pack.

Canonicalized distribution inventory, exact constraints for the resolver,
the file-level site-packages baseline and module-level provenance for the
vendor stack. Provenance matters because the runtime provides some modules
out-of-band (pip cannot see FlagTree's triton), so a pip-level zero-drift
check alone would miss a shadowing install.
"""

import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
import sys
import sysconfig
from pathlib import Path

from packaging.utils import canonicalize_name

# Mirror the runtime's container-start module resolution: PYTHONPATH
# (/opt/flagtree) precedes site-packages there, while build RUN steps do not
# inherit that ENV. Same position, same precedence as at container start.
if not os.environ.get("PYTHONPATH"):
    sys.path.insert(0, "/opt/flagtree")

evidence = Path("/opt/sglang-ci")

packages = {}
for dist in md.distributions():
    name = dist.metadata.get("Name")
    if name:
        packages[canonicalize_name(name)] = dist.version
with (evidence / "runtime-distributions-before.json").open("w") as f:
    json.dump(packages, f, indent=2, sort_keys=True)
with (evidence / "runtime-constraints.txt").open("w") as f:
    for name, version in sorted(packages.items()):
        f.write(f"{name}=={version}\n")

# File-level baseline for the runtime's site-packages: every file's SHA256.
# The post-install gate requires every baseline file to still exist with the
# identical hash, so an in-place overwrite by any closure wheel is caught
# even when pip metadata (or an import origin) stays unchanged.
purelib = Path(sysconfig.get_paths()["purelib"])
manifest = {}
for path in sorted(purelib.rglob("*")):
    if path.is_file():
        manifest[str(path.relative_to(purelib))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
with (evidence / "site-files-before.sha256").open("w") as f:
    for relative, digest in manifest.items():
        f.write(f"{digest}  {relative}\n")
print(f"site-packages baseline: {len(manifest)} files hashed under {purelib}")


def probe_module(module):
    # find_spec resolves without executing the module: no device libraries
    # are touched, so the probe is meaningful on a GPU-less builder.
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        return {"found": False}
    return {"found": True, "origin": spec.origin}


flagtree = Path("/opt/flagtree")
provenance = {
    "_context": {
        "pythonpath": os.environ.get("PYTHONPATH"),
        "flagtree_dir_present": flagtree.is_dir(),
        "flagtree_entries": (
            sorted(entry.name for entry in flagtree.iterdir())
            if flagtree.is_dir()
            else None
        ),
    },
    **{
        module: probe_module(module)
        for module in ("torch", "triton", "flag_gems", "flagcx", "flagtree")
    },
}
with (evidence / "module-provenance-before.json").open("w") as f:
    json.dump(provenance, f, indent=2, sort_keys=True)
print(json.dumps(provenance, indent=2, sort_keys=True))
