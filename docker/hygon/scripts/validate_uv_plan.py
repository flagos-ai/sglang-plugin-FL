#!/usr/bin/env python3
"""Validate the uv two-wheel dry-run plan against the runtime baseline.

Reads the uv ``--dry-run`` output (``Would install name==version ...``)
from stdin, turns it into the additions audit record, and fails when the
plan touches any runtime distribution. External platform components (the
excluded triton) never appear in a uv plan at all.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from packaging.utils import canonicalize_name

_WOULD_INSTALL_RE = re.compile(r"^Would install (.+)$", re.MULTILINE)

evidence = Path("/opt/sglang-ci")
plan_text = sys.stdin.read()
before = json.loads((evidence / "runtime-distributions-before.json").read_text())

planned: dict[str, str] = {}
for group in _WOULD_INSTALL_RE.findall(plan_text):
    for token in group.split():
        name, sep, version = token.partition("==")
        if not sep:
            raise SystemExit(f"FATAL: unparsable plan token {token!r}")
        planned[canonicalize_name(name)] = version

if not planned:
    raise SystemExit("FATAL: the dry-run plan is empty")

conflicts = sorted(
    f"{name}: runtime={before[name]} planned={version}"
    for name, version in planned.items()
    if name in before
)
if conflicts:
    print("FATAL: the plan replaces runtime distributions:")
    for line in conflicts:
        print(f"  ! {line}")
    sys.exit(1)

(evidence / "sglang-resolver-additions.txt").write_text(
    "\n".join(f"{name}=={version}" for name, version in sorted(planned.items())) + "\n"
)
print(
    f"resolver plans {len(planned)} new distributions "
    "(recorded in sglang-resolver-additions.txt)"
)
for name, version in sorted(planned.items()):
    print(f"  + {name}=={version}")
