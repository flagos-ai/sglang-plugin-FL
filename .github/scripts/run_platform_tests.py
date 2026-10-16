#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.

"""Dispatch platform tests while preserving the test runner's command line."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def build_command(argv: list[str], repo_root: Path | None = None) -> list[str]:
    """Select an optional platform hook without consuming its arguments."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--platform", required=True)
    args, _ = parser.parse_known_args(argv)
    if re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", args.platform) is None:
        parser.error(
            "--platform must be a platform id containing letters, digits, '_' or '-'"
        )

    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]
    hook = root / ".github" / "scripts" / args.platform / "run_tests.py"
    entrypoint = hook if hook.is_file() else root / "tests" / "run.py"
    return [sys.executable, str(entrypoint), *argv]


def main(argv: list[str] | None = None) -> None:
    command = build_command(list(sys.argv[1:] if argv is None else argv))
    # Windows' exec emulation can return success before the child exits.
    if sys.platform == "win32":
        raise SystemExit(subprocess.call(command))
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
