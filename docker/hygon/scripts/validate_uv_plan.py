#!/usr/bin/env python3
"""Validate the uv two-wheel dry-run plan against the runtime baseline.

Parses the REAL uv 0.11.30 ``--dry-run`` output (captured with stderr
included, which is where uv writes the plan):

    Resolved N packages in Xms
    Would download M packages
    Would install K packages
     + name==version                 (index packages)
     + name @ file:///tmp/artifacts/... (the staged local wheels)

Turns the plan into the additions audit record (the staged wheels
contribute the version read from their own METADATA) and fails when the
plan touches any runtime distribution. External platform components
(the excluded triton) never appear in a uv plan at all.

Usage: validate_uv_plan.py <uv-dry-run-output.txt> [evidence-dir]

The evidence dir (default /plan, the disposable plan dir mounted by
build_two_wheel.sh) must carry runtime-distributions-before.json; the
additions audit record is written next to it.
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

from packaging.utils import canonicalize_name

# ' + name==version' (index package) or ' + name @ file://...' (local wheel).
_PLAN_LINE_RE = re.compile(r"^\s*\+\s+(?P<token>.+)$", re.MULTILINE)
_WHEEL_VERSION_RE = re.compile(r"^Version:\s*(.+)$", re.MULTILINE)


def parse_plan(plan_text: str) -> dict[str, str]:
    """Map canonical name -> version from uv dry-run '+ ...' plan lines."""
    planned: dict[str, str] = {}
    for token in _PLAN_LINE_RE.findall(plan_text):
        if " @ " in token:
            # A staged local wheel: read the version straight from its
            # METADATA so the audit record carries a real version.
            name, _, location = token.partition(" @ ")
            wheel = Path(location.replace("file://", "").split("#")[0])
            with zipfile.ZipFile(wheel) as archive:
                dist_info = next(
                    n for n in archive.namelist() if n.endswith(".dist-info/METADATA")
                )
                metadata = archive.read(dist_info).decode("utf-8")
            match = _WHEEL_VERSION_RE.search(metadata)
            if match is None:
                raise SystemExit(f"FATAL: no Version in {wheel}'s METADATA")
            planned[canonicalize_name(name)] = match.group(1).strip()
            continue
        name, sep, version = token.partition("==")
        if not sep:
            raise SystemExit(f"FATAL: unparsable plan token {token!r}")
        planned[canonicalize_name(name)] = version
    if not planned:
        raise SystemExit(
            "FATAL: the dry-run plan is empty (no '+ ...' lines) - capture "
            "uv output with stderr included"
        )
    return planned


def main() -> int:
    if len(sys.argv) not in (2, 3):
        raise SystemExit(
            "usage: validate_uv_plan.py <uv-dry-run-output.txt> [evidence-dir]"
        )
    plan_text = Path(sys.argv[1]).read_text()
    evidence = Path(sys.argv[2]) if len(sys.argv) == 3 else Path("/plan")
    before = json.loads((evidence / "runtime-distributions-before.json").read_text())

    planned = parse_plan(plan_text)

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
        "\n".join(f"{name}=={version}" for name, version in sorted(planned.items()))
        + "\n"
    )
    print(
        f"resolver plans {len(planned)} new distributions "
        "(recorded in sglang-resolver-additions.txt)"
    )
    for name, version in sorted(planned.items()):
        print(f"  + {name}=={version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
