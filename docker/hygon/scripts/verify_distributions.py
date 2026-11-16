#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Post-install gate 1: distribution-level zero drift.

Every runtime distribution must survive with the exact same version and the
installed additions must match the reviewed resolver plan.
"""

import importlib.metadata as md
import json
import os
import sys
from pathlib import Path

from packaging.utils import canonicalize_name

# Same container-start resolution mirroring as the baseline capture above:
# the runtime's flagtree dist-info lives under /opt/flagtree, which is on
# sys.path via PYTHONPATH at container start.
if not os.environ.get("PYTHONPATH"):
    sys.path.insert(0, "/opt/flagtree")

evidence = Path("/opt/sglang-ci")
before = json.loads((evidence / "runtime-distributions-before.json").read_text())
after = {}
for dist in md.distributions():
    name = dist.metadata.get("Name")
    if name:
        after[canonicalize_name(name)] = dist.version
with (evidence / "runtime-distributions-after.json").open("w") as f:
    json.dump(after, f, indent=2, sort_keys=True)
with (evidence / "runtime-freeze-after.txt").open("w") as f:
    for name in sorted(after):
        f.write(f"{name}=={after[name]}\n")

drift = {n: (v, after.get(n)) for n, v in before.items() if after.get(n) != v}
removed = sorted(set(before) - set(after))
added = sorted(set(after) - set(before))
failed = False
(evidence / "sglang-additions-installed.txt").write_text(
    "\n".join(f"{name}=={after[name]}" for name in added) + "\n"
)
planned = set(
    (evidence / "sglang-resolver-additions.txt").read_text().split()
)
installed = set(f"{name}=={after[name]}" for name in added)
print(f"{len(added)} distributions added; runtime baseline size {len(before)}")
if installed != planned:
    failed = True
    print("FATAL: installed additions differ from the reviewed resolver plan:")
    for line in sorted(installed - planned):
        print(f"  + {line}")
    for line in sorted(planned - installed):
        print(f"  - {line}")
if drift:
    failed = True
    print("FATAL: runtime distribution versions drifted:")
    for name, (was, now) in sorted(drift.items()):
        print(f"  ! {name}: runtime={was} now={now}")
if removed:
    failed = True
    print("FATAL: runtime distributions removed: " + ", ".join(removed))
sys.exit(1 if failed else 0)
