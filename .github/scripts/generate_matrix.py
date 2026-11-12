#!/usr/bin/env python3
# Copyright (c) 2025 BAAI. All rights reserved.

"""Generate test matrices from a platform config.

Reads ``tests/platforms/<platform>.yaml``, expands the per-device test
definitions into flat JSON arrays, and writes them to ``$GITHUB_OUTPUT``
so that downstream GitHub Actions jobs can use ``fromJson()`` to fan out.

Usage (in a workflow step)::

    - id: matrix
      run: |
        python .github/scripts/generate_matrix.py \
          --platform ${{ inputs.platform }} \
          --ci-stage pr     # or nightly / weekly

Outputs:
    e2e        — JSON array of ``{task, device, cases, timeout, image, runner_labels,
                 container_options}`` objects grouped by (task, device); hardware fields are
                 the device's effective spec resolved from .github/configs/<platform>.yml.
    functional — JSON array of ``{device, image, runner_labels, container_options}``, one
                 per device that declares a functional test block.
    benchmark  — JSON array of the same shape, one per device with benchmark
                 enabled for the CI stage.

CI stage semantics (ported from vllm-plugin-FL #581)
---------------------------------------------------
pr
    E2E uses the ``pr`` sub-key of each device's ``tests.e2e`` block.
    PR cases are the subset declared by each platform YAML.
    Skips devices with no ``pr`` key defined.

nightly / weekly (default)
    E2E uses the ``nightly`` key.  ``weekly`` falls back to ``nightly``
    if no ``weekly`` key is defined in the YAML.
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
CONFIGS_DIR = REPO_ROOT / ".github" / "configs"

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


def load_hw_config(platform: str) -> dict:
    """Load the platform CI hardware config (.github/configs/<platform>.yml).

    Returns an empty dict when the file is absent so that matrix generation
    still works for platforms without a hardware config.
    """
    path = CONFIGS_DIR / f"{platform}.yml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve_device_hw(hw_cfg: dict, device: str) -> dict:
    """Effective hardware spec for *device*.

    Platform-level defaults (ci_image, runner_labels, container_options, ...)
    are merged with this device's ``devices.<device>`` overrides, if any.
    Concrete values are always attached — never missing keys or nulls —
    because GitHub's ``toJson(<missing>)`` yields the truthy string 'null',
    which would defeat ``a || b`` fallbacks in the workflow expressions.
    """
    base = {k: v for k, v in hw_cfg.items() if k != "devices"}
    overrides = (hw_cfg.get("devices") or {}).get(device) or {}
    merged = {**base, **overrides}
    return {
        "image": merged.get("ci_image"),
        "runner_labels": merged.get("runner_labels"),
        "container_options": merged.get("container_options") or "",
    }


def build_e2e_matrix(
    config: dict,
    devices: list[str],
    unsupported: list[str],
    ci_stage: str = "nightly",
    hw_cfg: dict | None = None,
) -> list[dict]:
    """End-to-end model tests (inference, serving, concurrent).

    These are slow and require model files. Entries are grouped by
    (task, device) so all model/case combos for the same group run in one job.

    The ``e2e`` block in each device's YAML is keyed by CI stage
    (``pr`` / ``nightly`` / ``weekly``).  In ``weekly`` stage, falls back
    to ``nightly`` if no ``weekly`` key is defined.
    """
    matrix: list[dict] = []

    for device in devices:
        device_cfg = config.get(device, {})
        tests = device_cfg.get("tests", {})
        e2e = tests.get("e2e", {})

        if ci_stage == "pr":
            # PR stage: read the 'pr' sub-key
            stage_block = e2e.get("pr", {})
            if not stage_block:
                continue
        else:
            # nightly/weekly: read the matching stage key; weekly falls
            # back to nightly
            stage_block = e2e.get(ci_stage) or e2e.get("nightly", {})

        for task, models in stage_block.items():
            if not isinstance(models, dict):
                continue
            cases: list[dict[str, str]] = []
            for model, case_list in models.items():
                # A bare scalar is accepted as a single case (same rule as
                # the runner's E2eTestConfig.get_cases).
                if not isinstance(case_list, list):
                    case_list = [case_list]
                for case in case_list:
                    full_name = f"{model}/{case}"
                    if any(token in full_name for token in unsupported):
                        continue
                    cases.append({"model": model, "case": case})

            if cases:
                entry = {
                    "task": task,
                    "device": device,
                    "cases": json.dumps(cases, separators=(",", ":")),
                    "timeout": DEFAULT_TIMEOUT.get(task, 60),
                }
                # Attach the per-device hardware spec (image / runner_labels /
                # container_options) so each job picks its runner directly
                # from the matrix entry.
                entry.update(resolve_device_hw(hw_cfg or {}, device))
                matrix.append(entry)

    return matrix


def build_functional_matrix(
    config: dict,
    devices: list[str],
    hw_cfg: dict | None = None,
) -> list[dict]:
    """One functional job per device that declares a functional block.

    Returns one entry per device with the hardware spec attached so the
    workflow does not need a separate runner-label lookup.  Devices whose
    functional block is empty (``functional: {}``) are skipped — used by
    profiles that share another device's silicon and runner pool.
    """
    entries = []
    for device in devices:
        section = config[device]
        functional = section.get("tests", {}).get("functional", {})
        if not functional:
            continue
        entry: dict = {"device": device}
        entry.update(resolve_device_hw(hw_cfg or {}, device))
        entries.append(entry)
    return entries


def build_benchmark_matrix(
    config: dict,
    devices: list[str],
    ci_stage: str = "nightly",
    hw_cfg: dict | None = None,
) -> list[dict]:
    """One benchmark job per device with benchmark enabled for this stage.

    The ``benchmark`` block is stage-keyed like ``e2e`` (``pr`` /
    ``nightly``; ``weekly`` falls back to ``nightly``).  Devices without an
    enabled stage block produce no entry, so their benchmark job is skipped
    instead of running an empty pass.
    """
    entries = []
    for device in devices:
        section = config[device]
        benchmark = section.get("tests", {}).get("benchmark", {})
        if not benchmark:
            continue
        stage_cfg = benchmark.get(ci_stage) or benchmark.get("nightly", {})
        if not stage_cfg or not stage_cfg.get("enabled", False):
            continue
        entry: dict = {"device": device}
        entry.update(resolve_device_hw(hw_cfg or {}, device))
        entries.append(entry)
    return entries


def load_changed_files(path: str) -> list[str]:
    """Load list of changed file paths from a text file (one per line)."""
    try:
        return Path(path).read_text().strip().splitlines()
    except (OSError, IOError):
        return []


# Any changed file starting with one of these prefixes triggers a full e2e run.
_FULL_RUN_PREFIXES: list[str] = [
    "sglang_fl/",
    "pyproject.toml",
    "tests/e2e_tests/",
    "tests/utils/",
    "tests/run.py",
    "tests/platforms/",
    "tests/conftest.py",
    ".github/",
]


def filter_e2e_by_changes(matrix: list[dict], changed: list[str]) -> list[dict]:
    """Narrow the e2e matrix based on changed files (PR smart-skip).

    Rules:
    - If *any* changed file matches a full-run prefix -> return all entries.
    - If all changes are under ``tests/models/<model>/`` -> keep only entries
      whose cases reference an affected model.
    - Unknown paths (not matching any known prefix) -> full run for safety.
    """
    for f in changed:
        if any(f.startswith(p) for p in _FULL_RUN_PREFIXES):
            print("[filter] Core/CI file changed -> full e2e run")
            return matrix

    affected_models: set[str] = set()
    for f in changed:
        if f.startswith("tests/models/"):
            parts = f.split("/")
            if len(parts) >= 3:
                affected_models.add(parts[2])
        else:
            # Unknown path outside known safe-to-ignore dirs -> full run
            print(f"[filter] Unknown path '{f}' -> full e2e run")
            return matrix

    if not affected_models:
        return matrix

    print(f"[filter] Only model configs changed, affected: {sorted(affected_models)}")

    filtered: list[dict] = []
    for entry in matrix:
        cases = json.loads(entry["cases"])
        kept = [c for c in cases if c["model"] in affected_models]
        if kept:
            filtered.append({**entry, "cases": json.dumps(kept, separators=(",", ":"))})
    return filtered


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
    parser.add_argument(
        "--ci-stage",
        default="nightly",
        choices=["pr", "nightly", "weekly"],
        help="CI stage: 'pr' uses the pr subset, 'nightly'/'weekly' use full cases.",
    )
    args = parser.parse_args()

    config = load_platform(args.platform)
    devices = get_device_sections(config)
    unsupported = config.get("unsupported_features", []) or []

    ci_stage = args.ci_stage

    # Hardware config from .github/configs/<platform>.yml: each device
    # inherits all platform-level defaults (ci_image, runner_labels,
    # container_options, ...) and overrides any field it declares under
    # devices:. Platforms without a devices: map resolve every device to the
    # platform defaults.
    hw_cfg = load_hw_config(args.platform)

    e2e_matrix = build_e2e_matrix(
        config, devices, unsupported, ci_stage=ci_stage, hw_cfg=hw_cfg
    )
    functional_matrix = build_functional_matrix(config, devices, hw_cfg=hw_cfg)
    benchmark_matrix = build_benchmark_matrix(
        config, devices, ci_stage=ci_stage, hw_cfg=hw_cfg
    )

    # Apply PR smart-skip filtering when changed files are provided. No CI
    # caller currently passes --changed-files (the platform-level diff router
    # in detect_platforms.py covers PR narrowing); the flag stays available
    # for local use. In pr stage the filter is skipped regardless.
    if args.changed_files and ci_stage != "pr":
        changed = load_changed_files(args.changed_files)
        if changed:
            e2e_matrix = filter_e2e_by_changes(e2e_matrix, changed)

    e2e_json = json.dumps(e2e_matrix, separators=(",", ":"))
    functional_json = json.dumps(functional_matrix, separators=(",", ":"))
    benchmark_json = json.dumps(benchmark_matrix, separators=(",", ":"))

    set_output("e2e", e2e_json)
    set_output("functional", functional_json)
    set_output("benchmark", benchmark_json)

    # Human-readable summary for CI logs
    print(f"Platform:    {args.platform}")
    print(f"Stage:       {ci_stage}")
    print(f"Devices:     {devices}")
    print(f"Functional:  {len(functional_matrix)} job(s)")
    for entry in functional_matrix:
        print(f"  - device={entry['device']} labels={entry.get('runner_labels')}")
    print(f"E2E:         {len(e2e_matrix)} job(s)")
    for entry in e2e_matrix:
        case_list = json.loads(entry["cases"])
        print(
            f"  - {entry['task']} (device={entry['device']}, "
            f"timeout={entry['timeout']}m, {len(case_list)} case(s))"
        )
        for c in case_list:
            print(f"      {c['model']}/{c['case']}")
    print(f"Benchmark:   {len(benchmark_matrix)} job(s)")
    for entry in benchmark_matrix:
        print(f"  - device={entry['device']} labels={entry.get('runner_labels')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
