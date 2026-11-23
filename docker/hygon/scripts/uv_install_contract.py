#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Print the uv install arguments of a Hygon runtime containerfile, one per
line, with the index ARG defaults substituted.

This lets other consumers run a containerfile's exact install command without
copying it by hand: docker/hygon/build_two_wheel.sh generates its dry-run
gate from it (the same arguments plus --dry-run) and
docker/hygon/no_excludes_gate.sh drops --excludes for its second plan. The
containerfile itself stays the single source of truth and the two-wheel
contract unit tests keep the consumers honest.

``--run-options`` prints the ``docker run`` options that reproduce the
install RUN's own build options - its bind mounts (resolved inside
``--context``) and its network mode - so a dry-run container sees exactly
the mounts and the network the recipe declares.

Usage: uv_install_contract.py [<containerfile>]
       uv_install_contract.py --run-options --context DIR [<containerfile>]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / "runtime-0.5.18.containerfile"
MARKER = "/usr/local/bin/uv pip install"
RUN_OPTION_PREFIXES = ("--mount=", "--network=")


def install_arguments(text: str) -> list[str]:
    args = {
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


def run_build_options(text: str) -> list[str]:
    """The build options (--mount=..., --network=...) of the RUN instruction
    that carries the install command."""
    lines = text.split(MARKER, 1)[0].split("\n")
    start = max(i for i, line in enumerate(lines) if line.lstrip().startswith("RUN"))
    tokens = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("RUN"):
            stripped = stripped[len("RUN") :].strip()
        tokens.extend(stripped.rstrip("\\").split())
    return [token for token in tokens if token.startswith(RUN_OPTION_PREFIXES)]


def run_options(text: str, context: Path | str | None = None) -> list[str]:
    """``docker run`` options reproducing the install RUN's build options:
    every read-only or writable bind mount of the recipe becomes a volume of
    the build context, and ``--network=none`` becomes an isolated container."""
    options: list[str] = []
    for option in run_build_options(text):
        key, _, value = option.partition("=")
        if key == "--network":
            if value == "none":
                options += ["--network", "none"]
            elif value != "default":
                raise SystemExit(f"FATAL: unsupported RUN network mode {value!r}")
            continue
        if context is None:
            raise SystemExit(
                "FATAL: resolving the recipe's bind mounts needs --context DIR"
            )
        fields = dict(part.split("=", 1) for part in value.split(",") if "=" in part)
        if fields.get("type") != "bind":
            raise SystemExit(f"FATAL: only bind mounts are supported: {option}")
        access = "ro" if "readonly" in value.split(",") else "rw"
        options += [
            "--volume",
            f"{Path(context) / fields['source']}:{fields['target']}:{access}",
        ]
    return options


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="uv install arguments of a Hygon runtime containerfile"
    )
    parser.add_argument(
        "--run-options",
        action="store_true",
        help="print docker run options reproducing the install RUN's mounts/network",
    )
    parser.add_argument(
        "--context",
        type=Path,
        help="build context directory the recipe's bind mounts resolve against",
    )
    parser.add_argument("containerfile", nargs="?", type=Path, default=DEFAULT)
    args = parser.parse_args(argv)
    text = args.containerfile.read_text(encoding="utf-8")
    arguments = (
        run_options(text, args.context) if args.run_options else install_arguments(text)
    )
    for argument in arguments:
        sys.stdout.write(argument + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
