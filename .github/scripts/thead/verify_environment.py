#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Prove the installed-wheel contract of the T-Head two-wheel CI image.

setup.sh runs this program twice, around the pinned test-runner install:

  --capture-runtime-snapshot FILE   record the distribution inventory
  --verify --runtime-evidence FILE  re-read it and check the contract

Only plugin-owned facts are asserted: the SGLang and sglang_fl wheels the
image installs, the entry points SGLang resolves them by, no standalone
triton distribution, and the semantic sgl_kernel placeholder. What the
Runtime owns (versions, compiler slots) is printed, never asserted.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import sys

EXPECTED_WHEELS = {"sglang": "0.5.18", "sglang-fl": "0.1.0"}
# Exact (name, value): SGLang loads the plugin through these entry points, so
# a value typo that still parses would silently break activation.
EXPECTED_ENTRY_POINTS = {
    "sglang.srt.platforms": ("sglang_fl", "sglang_fl:activate_platform"),
    "sglang.srt.plugins": ("sglang_fl", "sglang_fl:load_plugin"),
}


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def runtime_inventory(exclude_root: Path | None = None) -> dict[str, str]:
    """Canonical distribution name -> version.

    Anything located under ``exclude_root`` (the checkout) is skipped so that
    a source tree can never pass for an installed wheel.
    """
    inventory = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        if exclude_root is not None:
            origin = Path(dist.locate_file("")).resolve()
            if origin == exclude_root or exclude_root in origin.parents:
                continue
        inventory[canonical(name)] = dist.version
    return inventory


def inventory_drift(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Runtime distributions that changed or disappeared; additions are not drift."""
    return sorted(
        f"{name}: {version} -> {after.get(name, '<REMOVED>')}"
        for name, version in before.items()
        if after.get(name) != version
    )


def entry_point_failures(groups: dict[str, dict[str, str]]) -> list[str]:
    return [
        f"entry point {group}/{name} must be exactly {value!r}, "
        f"found {groups.get(group, {}).get(name)!r}"
        for group, (name, value) in EXPECTED_ENTRY_POINTS.items()
        if groups.get(group, {}).get(name) != value
    ]


def collected_entry_points() -> dict[str, dict[str, str]]:
    groups: dict[str, dict[str, str]] = {}
    for entry_point in metadata.entry_points():
        if entry_point.group in EXPECTED_ENTRY_POINTS:
            groups.setdefault(entry_point.group, {})[entry_point.name] = (
                entry_point.value
            )
    return groups


def wheel_origin_failures(plugin_root: Path) -> list[str]:
    """The plugin under test is the installed wheel, never the checkout."""
    import sglang_fl

    root = Path(metadata.distribution("sglang-fl").locate_file("")).resolve()
    module_file = Path(sglang_fl.__file__).resolve()
    if not module_file.is_relative_to(root):
        return [
            f"sglang_fl resolves from {module_file}, outside the installed "
            f"distribution root {root}"
        ]
    if module_file.is_relative_to(plugin_root):
        return [f"sglang_fl resolves from the checkout {plugin_root}: {module_file}"]
    return []


def placeholder_failures() -> list[str]:
    """Without an sgl_kernel distribution the plugin's placeholder carries its documented semantics."""
    try:
        version = metadata.version("sglang-kernel")
    except metadata.PackageNotFoundError:
        pass
    else:
        print(
            f"[thead] sgl_kernel: real distribution {version}, placeholder checks skipped"
        )
        return []

    from sglang_fl.dispatch.backends.vendor.thead.import_guard import (
        install_import_guards,
    )

    if "sgl_kernel" not in install_import_guards():
        return ["the sgl_kernel placeholder is not active"]
    # The import statement is the path SGLang takes; a plain attribute access
    # would only reach the generic call-fails-loudly placeholder.
    import sgl_kernel
    from sgl_kernel.utils import is_arch_support_pdl

    failures = []
    if is_arch_support_pdl() is not False:
        failures.append("sgl_kernel.utils.is_arch_support_pdl() must answer False")
    if hasattr(sgl_kernel, "scalar_type"):
        failures.append(
            "sgl_kernel.scalar_type must stay absent so SGLang's fallback takes over"
        )
    try:
        sgl_kernel.not_a_real_kernel()
    except RuntimeError:
        pass
    else:
        failures.append("an unknown sgl_kernel call must fail loudly")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-runtime-snapshot", type=Path, metavar="FILE")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--runtime-evidence", type=Path, metavar="FILE")
    parser.add_argument("--plugin-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    plugin_root = args.plugin_root.resolve()

    if args.capture_runtime_snapshot is not None:
        inventory = runtime_inventory(plugin_root)
        args.capture_runtime_snapshot.write_text(json.dumps(inventory, sort_keys=True))
        print(f"[thead] runtime snapshot: {len(inventory)} distributions")
        return 0
    if not args.verify or args.runtime_evidence is None:
        parser.error("--capture-runtime-snapshot or --verify --runtime-evidence")

    before = json.loads(args.runtime_evidence.read_text())
    after = runtime_inventory(plugin_root)

    import triton

    failures = inventory_drift(before, after)
    failures += [
        f"{name} must be {version}, found {after.get(name)}"
        for name, version in EXPECTED_WHEELS.items()
        if after.get(name) != version
    ]
    failures += entry_point_failures(collected_entry_points())
    failures += wheel_origin_failures(plugin_root)
    failures += placeholder_failures()
    if "triton" in after:
        failures.append(
            f"a standalone triton distribution is installed: {after['triton']}"
        )

    print(f"[thead] test tooling added: {sorted(set(after) - set(before))}")
    print(f"[thead] torch {after.get('torch')}, flag-gems {after.get('flag-gems')}")
    print(f"[thead] triton {triton.__version__} from {triton.__file__}")
    for failure in failures:
        print(f"::error:: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
