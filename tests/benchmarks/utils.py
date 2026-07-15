# Copyright (c) 2026 BAAI. All rights reserved.

"""Helpers for SGLang benchmark smoke tests."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


def load_benchmark_case() -> dict[str, Any]:
    """Load benchmark case config injected by tests/run.py."""
    raw = os.environ.get("FL_BENCHMARK_CASE")
    if not raw:
        raise RuntimeError("FL_BENCHMARK_CASE is not set")
    return json.loads(raw)


def to_cli_args(params: dict[str, Any], skip: set[str] | None = None) -> list[str]:
    """Convert snake_case parameter mapping to CLI args."""
    
    skip = skip or set()
    args: list[str] = []
    
    for key, value in params.items():
        if key in skip or value is None or value is False:
            continue
        
        flag = "--" + key.replace("_", "-")
        if value is True or value == "":
            args.append(flag)
        else:
            args.extend([flag, str(value)])
            
    return args


def run_command(command: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    print("[benchmark] Command:", " ".join(command))
    
    env = os.environ.copy()
    local_no_proxy = "127.0.0.1,localhost,::1"
    for key in ("NO_PROXY", "no_proxy"):
        current = env.get(key, "")
        env[key] = ",".join(filter(None, [current, local_no_proxy]))

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    print(result.stdout)
    print(result.stderr)
    return result

def read_last_jsonl(path: Path) -> dict[str, Any]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        raise AssertionError(f"No JSONL records found in {path}")
    return json.loads(lines[-1])

