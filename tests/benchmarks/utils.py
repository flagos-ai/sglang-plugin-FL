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


def _preload_plugin(command: list[str]) -> list[str]:
    """Run `python -m sglang.bench_one_batch` with the sglang_fl plugin loaded.

    bench_one_batch constructs ModelRunner directly and never triggers SGLang's
    plugin discovery (unlike launch_server), so vendor support registered by
    the plugin is missing in that interpreter — on Enflame GCU it dies with
    "Not supported device type: gcu" (DeviceConfig SUPPORTED_DEVICES). Loading
    the plugin in the same interpreter mirrors server-side behavior; on
    platforms whose vendor ships no patch module it changes nothing, and a
    preload failure degrades to the plain invocation instead of failing.
    """
    if command[1:3] != ["-m", "sglang.bench_one_batch"]:
        return command
    try:
        import sglang_fl  # noqa: F401
    except ImportError:
        return command
    wrapper = (
        "import sys, runpy\n"
        "try:\n"
        "    import sglang_fl\n"
        "    sglang_fl.load_plugin()\n"
        "except Exception as exc:\n"
        "    print(f'sglang_fl preload skipped: {exc!r}', file=sys.stderr)\n"
        "runpy.run_module(sys.argv.pop(1), run_name='__main__')"
    )
    return [command[0], "-c", wrapper] + command[2:]


def run_command(command: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    command = _preload_plugin(command)
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

