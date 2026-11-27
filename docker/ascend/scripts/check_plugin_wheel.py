#!/usr/bin/env python3
"""Acceptance checks for the staged sglang-plugin-FL wheel (Ascend contract).

Verifies the built wheel carries the two-wheel Ascend contract before it
enters any image build context:

  - METADATA Name/Version match the project,
  - the ``ascend`` optional dependency group is exactly the platform
    anchor plus the bounds the SGLang closure needs on a NumPy 1.x Runtime
    (see EXPECTED_EXTRA), compared as a normalized requirement SET,
  - the wheel declares none of numpy / triton: numpy is Runtime-owned and
    the Runtime provides the Triton compiler as the /opt/flagtree module
    tree,
  - both SGLang entry-point groups are registered with the EXACT
    ``(name, value)`` pairs -- a name-only match is not accepted,
  - the Ascend platform payload ships (vendor backend, op registrations,
    the sgl_kernel_npu placeholder and its fallbacks, dispatch configs),
  - no test files ship in the wheel.

Usage: check_plugin_wheel.py <wheel> [--repo-root DIR]
"""

import argparse
import re
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

EXPECTED_EXTRA = ("sglang==0.5.18", "pyarrow<26")
# Must never appear as a Requires-Dist: the extra is an honest dependency
# statement, not a pinning vehicle.
NOT_PLUGIN_DEPENDENCIES = ("numpy", "triton")
# Exact (name, value): the consumer (sglang's plugin/platform loader) calls
# the entry point's value, so a value typo that still parses would silently
# break activation.
EXPECTED_ENTRY_POINTS = {
    "sglang.srt.platforms": ("sglang_fl", "sglang_fl:activate_platform"),
    "sglang.srt.plugins": ("sglang_fl", "sglang_fl:load_plugin"),
}
EXPECTED_PAYLOAD = (
    "sglang_fl/dispatch/backends/vendor/ascend/ascend.py",
    "sglang_fl/dispatch/backends/vendor/ascend/register_ops.py",
    # Contract-critical: the import placeholder answers SGLang's module-level
    # sgl_kernel_npu imports, the fallbacks carry the GDN/MoE kernels, and
    # early.py installs both before SGLang imports its layers.
    "sglang_fl/dispatch/backends/vendor/ascend/import_guard.py",
    "sglang_fl/dispatch/backends/vendor/ascend/sgl_kernel_npu_fallbacks.py",
    "sglang_fl/dispatch/backends/vendor/ascend/early.py",
    "sglang_fl/dispatch/config/ascend.yaml",
    "sglang_fl/dispatch/config/ascend-cann921.yaml",
)


def project_version(repo_root: Path) -> str:
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', pyproject, re.MULTILINE)
    if match is None:
        raise SystemExit("cannot parse version from pyproject.toml")
    return match.group(1)


def parse_entry_points(entry_points_raw: str) -> dict[str, dict[str, str]]:
    """Map entry-point group -> {name: exact value}."""
    groups: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for line in entry_points_raw.splitlines():
        if line.startswith("[") and line.endswith("]"):
            current = groups.setdefault(line[1:-1], {})
        elif "=" in line.strip() and current is not None:
            name, _, value = line.partition("=")
            current[name.strip()] = value.strip()
    return groups


def requirement_name(entry: str) -> str:
    # PEP 503 canonical form, inlined so the acceptance check runs on
    # any host python3 without the packaging package installed.
    return re.split(r"\s|\[|;|<|>|=|!|~", entry.strip(), maxsplit=1)[0]


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def normalize_requirement(entry: str) -> str:
    """Canonical `name==version` / `name+clauses` form for exact matching.

    Strips the extra/environment markers setuptools appends and
    normalizes version-clause order/spacing, so
    `sglang==0.5.18; extra == "ascend"` and a pyproject-side
    `sglang==0.5.18` compare equal regardless of specifier rendering
    (`pyarrow<26,>=24` vs `pyarrow>=24,<26`).
    """
    requirement = entry.split(";", 1)[0].strip()
    if not requirement:
        return ""
    head = requirement_name(requirement)
    specifiers = requirement[len(head) :].strip()
    clauses = sorted(
        "".join(clause.split()) for clause in specifiers.split(",") if clause.strip()
    )
    return canonical(head) + ",".join(clauses) if clauses else canonical(head)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="checkout root for the pyproject comparison",
    )
    args = parser.parse_args()

    with zipfile.ZipFile(args.wheel) as archive:
        names = archive.namelist()
        dist_info = next(n for n in names if n.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(archive.read(dist_info).decode("utf-8"))
        entry_points_raw = archive.read(
            dist_info.replace("METADATA", "entry_points.txt")
        ).decode("utf-8")

    failures: list[str] = []

    name = metadata["Name"]
    if name != "sglang_fl":
        failures.append(f"Name must be sglang_fl, found {name!r}")

    version = metadata["Version"]
    expected_version = project_version(args.repo_root)
    if version != expected_version:
        failures.append(
            f"Version must match pyproject ({expected_version}), found {version!r}"
        )

    all_requirements = metadata.get_all("Requires-Dist") or []

    expected_set = {normalize_requirement(req) for req in EXPECTED_EXTRA}
    ascend_extra = [
        entry
        for entry in all_requirements
        if 'extra == "ascend"' in entry or "extra == 'ascend'" in entry
    ]
    actual_set = {normalize_requirement(entry) for entry in ascend_extra}
    if actual_set != expected_set:
        failures.append(
            f"ascend extra must be exactly {sorted(expected_set)} "
            f"(normalized, clause order folded), found {sorted(actual_set)}"
        )

    declared = sorted(
        entry
        for entry in all_requirements
        if canonical(requirement_name(entry)) in NOT_PLUGIN_DEPENDENCIES
    )
    if declared:
        failures.append(
            f"the plugin wheel must not declare "
            f"{list(NOT_PLUGIN_DEPENDENCIES)}, found: {declared}"
        )

    groups = parse_entry_points(entry_points_raw)
    for group, (expected_name, expected_value) in EXPECTED_ENTRY_POINTS.items():
        entries = groups.get(group, {})
        actual = entries.get(expected_name)
        if actual != expected_value:
            failures.append(
                f"entry-point group {group!r} must carry exactly "
                f"{expected_name} = {expected_value!r}, found {actual!r} "
                f"(all entries: {entries})"
            )

    missing_payload = [p for p in EXPECTED_PAYLOAD if p not in names]
    if missing_payload:
        failures.append(f"wheel lacks the Ascend payload: {missing_payload}")

    test_payload = sorted(n for n in names if n.startswith("tests/") or "/tests/" in n)
    if test_payload:
        failures.append(f"wheel ships test files, e.g. {test_payload[:3]}")

    if failures:
        for failure in failures:
            print(f"FATAL: {failure}", file=sys.stderr)
        return 1

    print(f"[wheel] {name}=={version}")
    print(
        f"[wheel] ascend extra: exact set {sorted(expected_set)}; "
        f"no {list(NOT_PLUGIN_DEPENDENCIES)} Requires-Dist"
    )
    print(
        "[wheel] entry points verified with exact (name, value): "
        + "; ".join(f"{g}: {n} = {v}" for g, (n, v) in EXPECTED_ENTRY_POINTS.items())
    )
    print("[wheel] Ascend payload present; no tests shipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
