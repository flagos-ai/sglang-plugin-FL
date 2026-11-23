#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Print the uv install arguments of runtime-0.5.18.containerfile, one per
line, with the index ARG defaults substituted.

This lets a second consumer (docker/hygon/no_excludes_gate.sh) run the
containerfile's exact install command without copying it by hand; the
containerfile itself stays the single source of truth and the two-wheel
contract unit tests keep comparing it against build_two_wheel.sh's dry-run.

Usage: uv_install_contract.py [<containerfile>]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / "runtime-0.5.18.containerfile"
MARKER = "/usr/local/bin/uv pip install"


def install_arguments(text: str) -> list[str]:
    args: dict[str, str] = {
        match.group(1): match.group(2)
        for match in re.finditer(r"^ARG (\w+)=(\S+)$", text, re.MULTILINE)
    }
    marker_line, after = text.split(MARKER, 1)[1].split("\n", 1)
    lines = [marker_line.rstrip("\\").strip()]
    # the RUN instruction continues while lines end with a backslash
    for line in after.splitlines():
        stripped = line.strip()
        continuation = stripped.endswith("\\")
        lines.append(stripped.rstrip("\\").strip())
        if not continuation:
            break
    joined = " ".join(lines)
    for name, value in args.items():
        joined = joined.replace(f'"${name}"', value)
    # the RUN shell strips the single quotes around the wheel requirements
    return [token.strip("'") for token in joined.split() if token]


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    for argument in install_arguments(path.read_text(encoding="utf-8")):
        sys.stdout.write(argument + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
