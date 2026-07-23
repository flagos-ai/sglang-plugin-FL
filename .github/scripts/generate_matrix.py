#!/usr/bin/env python3
# Copyright (c) 2025 BAAI. All rights reserved.

"""Generate test matrices from a platform config.

Reads ``tests/platforms/<platform>.yaml``, expands the per-device test
definitions into flat JSON arrays, and writes them to ``$GITHUB_OUTPUT``
so that downstream GitHub Actions jobs can use ``fromJson()`` to fan out.

Usage (in a workflow step)::

    - id: matrix
      run: python .github/scripts/generate_matrix.py --platform ${{ inputs.platform }}

Outputs:
    e2e  — JSON array of ``{task, device, cases, timeout}`` objects grouped by (task, device).
    unit — JSON array of ``{device, include, exclude}`` objects.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PLATFORMS_DIR = REPO_ROOT / "tests" / "platforms"

# YAML task key → test directory name
TASK_DIR_MAP: dict[str, str] = {
    "serve": "serving",
}

# Default timeout (minutes) per task category
DEFAULT_TIMEOUT: dict[str, int] = {
    "inference": 60,
    "serving": 60,
    "concurrent": 60,
}


def load_platform(platform: str) -> dict:
    """Load and return the platform YAML config."""
    path = PLATFORMS_DIR / f"{platform}.yaml"
    if not path.exists():
        print(f"::error::Platform config not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_task_dir(task_key: str) -> str:
    """Map a YAML task key to the corresponding test directory name."""
    return TASK_DIR_MAP.get(task_key, task_key)


def get_device_sections(config: dict) -> list[str]:
    """Return device section names present in the config.

    Device sections are top-level keys that have a nested ``tests`` mapping.
    """
    devices = []
    for key, value in config.items():
        if isinstance(value, dict) and "tests" in value:
            devices.append(key)
    return devices


def build_e2e_matrix(
    config: dict,
    devices: list[str],
    unsupported: list[str],
) -> list[dict]:
    """End-to-end model tests (inference, serving, concurrent).

    These are slow and require model files. Entries are grouped by
    (task, device) so all model/case combos for the same group run in one job.
    """
    matrix: list[dict] = []

    for device in devices:
        device_cfg = config.get(device, {})
        tests = device_cfg.get("tests", {})
        e2e = tests.get("e2e", {})

        for task, models in e2e.items():
            if not isinstance(models, dict):
                continue
            cases: list[dict[str, str]] = []
            for model, case_list in models.items():
                if not isinstance(case_list, list):
                    continue
                for case in case_list:
                    full_name = f"{model}/{case}"
                    if any(token in full_name for token in unsupported):
                        continue
                    cases.append({"model": model, "case": case})

            if cases:
                matrix.append(
                    {
                        "task": task,
                        "device": device,
                        "cases": json.dumps(cases, separators=(",", ":")),
                        "timeout": DEFAULT_TIMEOUT.get(task, 60),
                    }
                )

    return matrix


def build_unit_matrix(config: dict, devices: list[str]) -> list[dict]:
    """Unit test matrix — one entry per device with include/exclude patterns."""
    matrix: list[dict] = []

    for device in devices:
        device_cfg = config.get(device, {})
        tests = device_cfg.get("tests", {})
        unit = tests.get("unit", {})
        matrix.append(
            {
                "device": device,
                "include": unit.get("include", "*"),
                "exclude": json.dumps(unit.get("exclude", []), separators=(",", ":")),
            }
        )

    return matrix


def load_changed_files(path: str) -> list[str]:
    """Load list of changed file paths from a text file (one per line)."""
    try:
        return Path(path).read_text().strip().splitlines()
    except (OSError, IOError):
        return []


def filter_e2e_by_changes(matrix: list[dict], changed: list[str]) -> list[dict]:
    """Smart-skip: if only docs/CI/examples changed, skip e2e tests."""
    skip_prefixes = ("docs/", ".github/", "examples/", "README", "LICENSE")
    if all(any(f.startswith(prefix) for prefix in skip_prefixes) for f in changed):
        print("::notice::Only docs/CI files changed — skipping e2e tests")
        return []
    return matrix


def set_output(name: str, value: str) -> None:
    """Write a key=value pair to $GITHUB_OUTPUT (or print for local runs)."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"{name}<<EOF\n{value}\nEOF\n")
    else:
        print(f"{name}={value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True)
    parser.add_argument(
        "--changed-files",
        default=None,
        help="Path to a file listing changed paths (one per line)",
    )
    args = parser.parse_args()

    config = load_platform(args.platform)
    devices = get_device_sections(config)
    unsupported = config.get("unsupported_features", []) or []

    e2e_matrix = build_e2e_matrix(config, devices, unsupported)
    unit_matrix = build_unit_matrix(config, devices)

    # Apply PR smart-skip filtering when changed files are provided
    if args.changed_files:
        changed = load_changed_files(args.changed_files)
        if changed:
            e2e_matrix = filter_e2e_by_changes(e2e_matrix, changed)

    e2e_json = json.dumps(e2e_matrix, separators=(",", ":"))
    unit_json = json.dumps(unit_matrix, separators=(",", ":"))

    set_output("e2e", e2e_json)
    set_output("unit", unit_json)

    # Human-readable summary for CI logs
    print(f"Platform:    {args.platform}")
    print(f"Devices:     {devices}")
    print(f"E2E:         {len(e2e_matrix)} job(s)")
    for entry in e2e_matrix:
        case_list = json.loads(entry["cases"])
        print(
            f"  - {entry['task']} (device={entry['device']}, "
            f"timeout={entry['timeout']}m, {len(case_list)} case(s))"
        )
        for c in case_list:
            print(f"      {c['model']}/{c['case']}")
    print(f"Unit:        {len(unit_matrix)} config(s)")
    for entry in unit_matrix:
        print(
            f"  - device={entry['device']} "
            f"include={entry['include']} exclude={entry['exclude']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
