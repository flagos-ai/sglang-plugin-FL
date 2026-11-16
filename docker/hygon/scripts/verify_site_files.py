#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Post-install gate 1b: file-level zero drift on site-packages.

Every baseline site-packages file must still exist with the identical
SHA256 (catches in-place overwrites that pip metadata cannot see).
"""

import hashlib
import sys
import sysconfig
from pathlib import Path

evidence = Path("/opt/sglang-ci")
purelib = Path(sysconfig.get_paths()["purelib"])
baseline = {}
for line in (evidence / "site-files-before.sha256").read_text().splitlines():
    digest, relative = line.split("  ", 1)
    baseline[relative] = digest

missing = []
changed = []
for relative, expected in baseline.items():
    path = purelib / relative
    if not path.is_file():
        missing.append(relative)
    elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        changed.append(relative)
if missing or changed:
    print(
        f"FATAL: site-packages baseline drifted: {len(missing)} missing, "
        f"{len(changed)} content-changed"
    )
    for item in missing[:20]:
        print(f"  - missing: {item}")
    for item in changed[:20]:
        print(f"  ! changed: {item}")
    sys.exit(1)
print(
    f"site-packages file baseline unchanged "
    f"({len(baseline)} files re-hashed identical)"
)
