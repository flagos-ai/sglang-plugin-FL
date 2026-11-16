#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate the pip dry-run resolver plan against the runtime baseline.

Any runtime distribution in the install plan fails the build: a different
version is a hard conflict, an identical version is a reinstall that needs
explicit review. New (absent) distributions are the SGLang closure; they are
recorded for review but allowed.
"""

import json
import sys
from pathlib import Path

from packaging.utils import canonicalize_name

evidence = Path("/opt/sglang-ci")
before = json.loads((evidence / "runtime-distributions-before.json").read_text())
report = json.loads((evidence / "sglang-resolver-plan.json").read_text())

additions, same_version, conflicts = [], [], []
for item in report.get("install", []):
    metadata = item.get("metadata", {})
    name = canonicalize_name(metadata.get("name", ""))
    version = metadata.get("version", "")
    if name not in before:
        additions.append((name, version))
    elif version == before[name]:
        same_version.append((name, version))
    else:
        conflicts.append((name, before[name], version))

(evidence / "sglang-resolver-additions.txt").write_text(
    "\n".join(f"{name}=={version}" for name, version in sorted(additions)) + "\n"
)
print(f"resolver plans {len(additions)} new distributions "
      "(recorded in sglang-resolver-additions.txt)")
for name, version in sorted(additions):
    print(f"  + {name}=={version}")

failed = False
if same_version:
    failed = True
    print("FATAL: resolver plans to reinstall runtime distributions "
          "(same version) - explicit review required:", flush=True)
    for name, version in sorted(same_version):
        print(f"  = {name}=={version}")
if conflicts:
    failed = True
    print("FATAL: resolver plans to replace runtime distributions "
          "with different versions:", flush=True)
    for name, old, new in sorted(conflicts):
        print(f"  ! {name}: runtime={old} planned={new}")
if failed:
    sys.exit(1)
