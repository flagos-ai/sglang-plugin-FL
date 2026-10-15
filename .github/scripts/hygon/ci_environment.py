#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Preserve the HCU allocation and disable CUDA-only Torch extension autoload."""

from __future__ import annotations

import argparse
from importlib import metadata
import os
from pathlib import Path
import subprocess

EXPECTED_SGLANG = "0.5.18"
CI_DISTRIBUTIONS = ("pytest", "pytest-timeout", "pyyaml", "Pillow", "requests")


def allocation_mask(environ):
    """HIP is authoritative; never guess relative ROCR/CUDA remapping."""
    if "HIP_VISIBLE_DEVICES" in environ:
        value = environ["HIP_VISIBLE_DEVICES"]
    else:
        legacy = [
            environ[key]
            for key in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
            if key in environ
        ]
        if len(legacy) == 2 and legacy[0] != legacy[1]:
            raise RuntimeError(
                "Ambiguous ROCR_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES allocations; "
                "provide the intended HIP_VISIBLE_DEVICES explicitly"
            )
        value = legacy[0] if legacy else None
    if value is not None and any(
        character in value for character in ("\n", "\r", "\0")
    ):
        raise RuntimeError("GPU allocation must be a single environment-variable value")
    return value


def normalized_environment(environ):
    result = dict(environ)
    mask = allocation_mask(environ)
    result.pop("CUDA_VISIBLE_DEVICES", None)
    result.pop("ROCR_VISIBLE_DEVICES", None)
    result["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    if mask is not None:
        result["HIP_VISIBLE_DEVICES"] = mask
    return result


def persist_for_actions(environ):
    destination = environ.get("GITHUB_ENV")
    if destination:
        # Empty legacy masks can hide every GPU. HCU test steps explicitly
        # unset them; HIP is the only persisted allocation.
        with Path(destination).open("a", encoding="utf-8") as output:
            output.write("TORCH_DEVICE_BACKEND_AUTOLOAD=0\n")
            if "HIP_VISIBLE_DEVICES" in environ:
                output.write(
                    "HIP_VISIBLE_DEVICES=" + environ["HIP_VISIBLE_DEVICES"] + "\n"
                )


def verify_image_metadata(version=metadata.version):
    actual = version("sglang")
    if actual != EXPECTED_SGLANG:
        raise RuntimeError(
            f"Hygon CI requires SGLang {EXPECTED_SGLANG}; image contains {actual}. "
            "The legacy main-branch 0.5.11 CI image cannot run this suite."
        )
    for package in CI_DISTRIBUTIONS:
        print(f"[hygon-ci] {package}={version(package)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-image", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    environ = normalized_environment(os.environ)
    if args.check_image:
        verify_image_metadata()
    persist_for_actions(environ)
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if command:
        return subprocess.run(command, env=environ, check=False).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
