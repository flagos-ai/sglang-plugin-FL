#!/usr/bin/env python3
"""Temporarily make the HCU clang invocation use DTK's device libraries."""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "# Hygon CI diagnostic: explicit DTK device-library path."
TRACE_MARKER = "# Hygon CI diagnostic: capture clang output-file behavior."
RETURN = "        return clang_args\n"
SUBPROCESS = "            result = subprocess.run(asm_command, check=True, capture_output=True, text=True)\n"


def enable_fallback(compiler: Path, device_lib: Path) -> None:
    if not device_lib.is_dir():
        raise RuntimeError(f"DTK device-library directory is absent: {device_lib}")
    for required in ("ocml.bc", "ockl.bc"):
        if not (device_lib / required).is_file():
            raise RuntimeError(f"DTK device library is absent: {device_lib / required}")

    source = compiler.read_text(encoding="utf-8")
    if MARKER in source and TRACE_MARKER in source:
        print(f"[hygon-ci] FlagTree JIT diagnostics already enabled in {compiler}")
        return
    if source.count(RETURN) != 1:
        raise RuntimeError(
            f"unexpected FlagTree compiler shape: found {source.count(RETURN)} clang returns"
        )

    replacement = (
        f"        {MARKER}\n"
        "        # FlagTree 0.7.0 otherwise asks clang to discover this path and\n"
        "        # fails on the BW1000 runner despite ROCM_PATH=/opt/dtk. Also\n"
        "        # remove its disabled-option empty string before invoking clang.\n"
        "        clang_args = [arg for arg in clang_args if arg]\n"
        f'        clang_args.append("--rocm-device-lib-path={device_lib}")\n' + RETURN
    )
    updated = source.replace(RETURN, replacement, 1)
    if source.count(SUBPROCESS) != 1:
        raise RuntimeError(
            "unexpected FlagTree compiler shape: expected one clang subprocess call"
        )
    traced_subprocess = (
        SUBPROCESS
        + "            print(f'[hygon-ci] clang argv: {asm_command!r}', flush=True)\n"
        + "            print(f'[hygon-ci] clang rc={result.returncode} output_exists={os.path.exists(asm_file)}', flush=True)\n"
        + "            print(f'[hygon-ci] clang stdout: {result.stdout}', flush=True)\n"
        + "            print(f'[hygon-ci] clang stderr: {result.stderr}', flush=True)\n"
    )
    updated = updated.replace(SUBPROCESS, traced_subprocess, 1)
    updated = updated.replace(
        "        # Hygon CI diagnostic: explicit DTK device-library path.\n",
        "        # Hygon CI diagnostic: explicit DTK device-library path.\n"
        f"        {TRACE_MARKER}\n",
        1,
    )
    compile(updated, str(compiler), "exec")
    compiler.write_text(updated, encoding="utf-8")
    print(
        "[hygon-ci] enabled diagnostic FlagTree device-library fallback "
        f"({device_lib}) in {compiler}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compiler",
        type=Path,
        default=Path("/opt/flagtree/triton/backends/hcu/compiler_hcu.py"),
    )
    parser.add_argument(
        "--device-lib",
        type=Path,
        default=Path("/opt/dtk/dcc/dccgcn/bitcode"),
    )
    args = parser.parse_args()
    enable_fallback(args.compiler, args.device_lib)


if __name__ == "__main__":
    main()
