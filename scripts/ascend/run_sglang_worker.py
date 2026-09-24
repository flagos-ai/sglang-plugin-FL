#!/usr/bin/env python3
"""Run a multi-node SGLang worker with strict 0.5.18 exit normalization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-schedulers",
        type=int,
        required=True,
        help="Number of local scheduler exits required for the known clean path.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Worker command, normally after a '--' separator.",
    )
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a worker command is required after '--'")
    if args.expected_schedulers <= 0:
        parser.error("--expected-schedulers must be positive")
    return args


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "examples"))

    from _multinode_worker import run_sglang_worker

    return run_sglang_worker(args.command, args.expected_schedulers)


if __name__ == "__main__":
    raise SystemExit(main())
