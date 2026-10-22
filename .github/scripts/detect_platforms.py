#!/usr/bin/env python3
# Copyright (c) 2025 BAAI. All rights reserved.

"""Detect which platforms to test in CI.

Priority:
  1. If .github/configs/platforms.yml exists, read it and return only
     platforms with ``enabled: true``, filtered by diff-based routing
     when ``--changed-files`` is provided.
  2. Otherwise, fall back to auto-scanning .github/configs/*.yml,
     excluding ``template`` and ``platforms`` (the registry file itself).

Diff-based routing logic (when --changed-files is given):
  - If any changed file matches ``global_trigger_paths`` → return all enabled
    platforms (full run). Paths carved out by a platform's
    ``exempt_from_global`` never count as global triggers — they route only
    via the owning platform's ``trigger_paths``.
  - Otherwise, return only platforms whose ``trigger_paths`` match at least
    one changed file.
  - Platforms with empty ``trigger_paths`` are always included when enabled.
  - Divergence from the vllm-plugin-FL original (#581): a changed file that
    matches no rule at all (not ignored, not global, not exempt, not any
    platform's trigger_paths) triggers a full run. A brand-new shared file
    that the lists above do not know about must not silently skip device
    tests; over-triggering only costs runner time.

Usage (in a workflow step)::

    - id: detect
      run: |
        python3 .github/scripts/detect_platforms.py \\
          --changed-files /tmp/changed_files.txt

Sets the GitHub Actions output ``platforms`` to a JSON array of platform
names, e.g. ``["nvidia", "ascend", "musa"]``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
REGISTRY_FILE = CONFIGS_DIR / "platforms.yml"

# Names to exclude when falling back to auto-scan
AUTO_SCAN_EXCLUDE = {"template", "platforms"}

# Paths that never influence routing. Keep in sync with the ``paths-ignore``
# block of .github/workflows/ci.yml — the workflow only skips entirely when
# ALL changed files are ignored, so mixed PRs still route on their code
# changes while doc-only files are dropped here.
IGNORED_SUFFIXES = (".md",)
IGNORED_PATHS = (
    "docs/",
    "examples/",
    "LICENSE",
    ".github/ISSUE_TEMPLATE/",
    ".github/PULL_REQUEST_TEMPLATE.md",
)


def _is_ignored(path: str) -> bool:
    if path.endswith(IGNORED_SUFFIXES):
        return True
    return any(path.startswith(p) or path == p for p in IGNORED_PATHS)


def _matches(path: str, rule: str) -> bool:
    """Directory rules (trailing '/') prefix-match; file rules match exactly
    (a file rule must not prefix-capture siblings like nvidia.yaml.bak)."""
    if rule.endswith("/"):
        return path.startswith(rule)
    return path == rule


def _validate_and_collect_exemptions(
    enabled: dict, global_paths: list[str]
) -> list[str]:
    """Collect exempt_from_global prefixes from enabled platforms.

    Guards two-list drift in both directions:
      error   - exempt entry missing from the same platform's trigger_paths
                (files matching it would route to NO platform -> exit 2).
      warning - trigger_path under a global prefix but not exempted
                (changes there trigger a full run -> over-trigger only).
      warning - exempt entry not under any global prefix (no-op / stale).
    """
    exempt_paths: list[str] = []
    for name, cfg in enabled.items():
        triggers: list[str] = cfg.get("trigger_paths", [])
        exempts: list[str] = cfg.get("exempt_from_global", [])
        for tp in triggers:
            if tp not in exempts and any(tp.startswith(gp) for gp in global_paths):
                print(
                    f"::warning::platform '{name}': trigger_path '{tp}' is under "
                    f"a global path but not in exempt_from_global — changes "
                    f"there will trigger a full run",
                    file=sys.stderr,
                )
        for ep in exempts:
            if ep not in triggers:
                print(
                    f"::error::platform '{name}': exempt_from_global entry "
                    f"'{ep}' is not in trigger_paths — files matching it "
                    f"would route to no platform",
                    file=sys.stderr,
                )
                sys.exit(2)
            if not any(ep.startswith(gp) or gp.startswith(ep) for gp in global_paths):
                print(
                    f"::warning::platform '{name}': exempt_from_global entry "
                    f"'{ep}' is not under any global path (no-op)",
                    file=sys.stderr,
                )
            exempt_paths.append(ep)
    return exempt_paths


def from_registry(changed_files: list[str] | None = None) -> list[str] | None:
    """Read platforms.yml and return enabled platform names, filtered by
    diff-based routing when changed_files is provided. Returns None if the
    file does not exist."""
    if not REGISTRY_FILE.exists():
        return None

    with open(REGISTRY_FILE) as f:
        data = yaml.safe_load(f)

    # A structurally invalid registry must fail the discover job (and thus
    # ci-gate) instead of returning [] with exit 0 — that would silently
    # skip every platform test while staying green. An intentionally empty
    # registry (platforms: {} or all enabled:false) is valid and returns [].
    if not isinstance(data, dict) or "platforms" not in data:
        print(
            "::error::platforms.yml exists but has no 'platforms' key",
            file=sys.stderr,
        )
        sys.exit(2)
    platforms = data["platforms"]
    if not isinstance(platforms, dict):
        print("::error::platforms.yml 'platforms' is not a mapping", file=sys.stderr)
        sys.exit(2)

    enabled = {
        name: cfg
        for name, cfg in platforms.items()
        if isinstance(cfg, dict) and cfg.get("enabled", False)
    }

    if not enabled:
        return []

    global_paths: list[str] = data.get("global_trigger_paths", [])
    exempt_paths = _validate_and_collect_exemptions(enabled, global_paths)

    # No diff filtering → return all enabled platforms (schedule/dispatch)
    if changed_files is None:
        return list(enabled)

    # Drop doc-only paths (synced with ci.yml paths-ignore) before routing.
    changed = [f for f in changed_files if not _is_ignored(f)]
    if not changed:
        return list(enabled)

    # Diff-based routing. Two keys, two independent questions:
    #   - exempt_from_global: vendor carve-out; a changed file under one of
    #     these prefixes does NOT count toward global_trigger_paths.
    #   - trigger_paths: answers only "does this platform run?" and has no
    #     effect on the global trigger.
    for f in changed:
        if any(_matches(f, ep) for ep in exempt_paths):
            continue  # vendor carve-out: not a global trigger
        for gp in global_paths:
            if _matches(f, gp):
                print(
                    f"[detect] '{f}' matches global path '{gp}' → all platforms",
                    file=sys.stderr,
                )
                return list(enabled)

    # Per-platform matching; also track which files matched any rule so an
    # entirely unknown path can fall back to a full run (safety default).
    routed: set[str] = {
        f for f in changed if any(_matches(f, ep) for ep in exempt_paths)
    }
    triggered: list[str] = []
    for name, cfg in enabled.items():
        trigger_paths: list[str] = cfg.get("trigger_paths", [])

        # Empty trigger_paths → always include when enabled
        if not trigger_paths:
            triggered.append(name)
            continue

        matched = False
        # Scan every changed file: an early exit after the first match would
        # leave the platform's other files unrecorded in `routed`, and the
        # unmatched-file fallback would then over-trigger a full run.
        for f in changed:
            for tp in trigger_paths:
                if _matches(f, tp):
                    print(
                        f"[detect] '{f}' matches '{tp}' → include platform '{name}'",
                        file=sys.stderr,
                    )
                    matched = True
                    routed.add(f)
                    break

        if matched:
            triggered.append(name)

    for f in changed:
        if f not in routed:
            print(
                f"[detect] '{f}' matches no routing rule → full run (safety default)",
                file=sys.stderr,
            )
            return list(enabled)

    return triggered


def from_auto_scan() -> list[str]:
    """Scan .github/configs/*.yml and return platform names (minus exclusions)."""
    if not CONFIGS_DIR.is_dir():
        print(f"::error::Configs directory not found: {CONFIGS_DIR}", file=sys.stderr)
        return []

    return sorted(
        p.stem for p in CONFIGS_DIR.glob("*.yml") if p.stem not in AUTO_SCAN_EXCLUDE
    )


def load_changed_files(path: str) -> list[str] | None:
    """Read a newline-delimited file of changed paths. Returns None if empty."""
    p = Path(path)
    if not p.exists():
        print(f"::warning::changed-files path not found: {path}", file=sys.stderr)
        return None
    lines = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    return lines or None


def set_output(name: str, value: str) -> None:
    """Write a key=value pair to $GITHUB_OUTPUT (or print for local runs)."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"{name}<<EOF\n{value}\nEOF\n")
    else:
        print(f"{name}={value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect CI platforms")
    parser.add_argument(
        "--changed-files",
        metavar="PATH",
        help="Newline-delimited file of changed paths for diff-based routing",
    )
    args = parser.parse_args(argv)

    changed: list[str] | None = None
    if args.changed_files:
        changed = load_changed_files(args.changed_files)
        if changed is not None:
            print(f"[detect] Changed files ({len(changed)}):", file=sys.stderr)
            for f in changed[:20]:
                print(f"  {f}", file=sys.stderr)
            if len(changed) > 20:
                print(f"  ... and {len(changed) - 20} more", file=sys.stderr)
        else:
            print("[detect] No changed files → full run", file=sys.stderr)

    platforms = from_registry(changed)

    if platforms is not None:
        source = "platforms.yml"
    else:
        print(
            "::notice::platforms.yml not found, falling back to auto-scan",
            file=sys.stderr,
        )
        platforms = from_auto_scan()
        source = "auto-scan"

    if not platforms:
        print("::warning::No platforms detected", file=sys.stderr)

    result = json.dumps(platforms)
    set_output("platforms", result)

    print(f"Source:     {source}")
    print(f"Platforms:  {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
