#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Post-install gate 3: vendor module provenance.

Every vendor module must still resolve to the exact same origin. The
image-level PYTHONPATH=/opt/flagtree makes the runtime's FlagTree tree the
authoritative resolution in every entrypoint (build steps included), and the
standalone triton wheel is removed from site-packages at install time, so
triton consistently resolves under /opt/flagtree before and after the
closure install; CI job-time checks re-assert that resolution.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

# Same container-start resolution mirroring as the baseline probe.
if not os.environ.get("PYTHONPATH"):
    sys.path.insert(0, "/opt/flagtree")

evidence = Path("/opt/sglang-ci")


def probe_module(module):
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        return {"found": False}
    return {"found": True, "origin": spec.origin}


after = {
    module: probe_module(module)
    for module in ("torch", "triton", "flag_gems", "flagcx", "flagtree")
}
with (evidence / "module-provenance-after.json").open("w") as f:
    json.dump(after, f, indent=2, sort_keys=True)

recorded = json.loads((evidence / "module-provenance-before.json").read_text())
context = recorded.pop("_context")
before = recorded

changed = {
    module: (before.get(module), after.get(module))
    for module in sorted(set(before) | set(after))
    if before.get(module) != after.get(module)
}
if changed:
    print("FATAL: vendor module provenance changed across the install:")
    for module, (was, now) in changed.items():
        print(f"  ! {module}: {was} -> {now}")
    print("--- recorded build context ---")
    print(json.dumps(context, indent=2, sort_keys=True))
    sys.exit(1)
print(
    "module provenance unchanged under container-start resolution order "
    "(triton resolves via /opt/flagtree before site-packages):"
)
print(json.dumps(after, indent=2, sort_keys=True))
