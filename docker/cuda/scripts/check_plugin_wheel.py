#!/usr/bin/env python3
"""Acceptance checks for the staged sglang-plugin-FL wheel.

Verifies the built wheel carries the two-wheel NVIDIA contract before it
enters any image build context:

  - METADATA Name/Version match the project,
  - the ``nvidia`` optional dependency group exists and carries the
    compatibility anchors (sglang pin, triton slot pin, runtime numpy),
  - both SGLang entry-point groups are registered,
  - sglang_fl/nvidia_runtime.py ships in the wheel,
  - no test files ship in the wheel.

Usage: check_plugin_wheel.py <wheel> [--repo-root DIR]
"""

import argparse
import re
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

EXPECTED_EXTRA_ANCHORS = (
    "sglang==0.5.18",
    "triton==3.7.1",
    "numpy==1.26.4",
)
EXPECTED_ENTRY_POINTS = {
    "sglang.srt.platforms": "sglang_fl",
    "sglang.srt.plugins": "sglang_fl",
}


def project_version(repo_root: Path) -> str:
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', pyproject, re.MULTILINE)
    if match is None:
        raise SystemExit("cannot parse version from pyproject.toml")
    return match.group(1)


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

    nvidia_extra = sorted(
        entry
        for entry in metadata.get_all("Requires-Dist") or []
        if 'extra == "nvidia"' in entry or "extra == 'nvidia'" in entry
    )
    if not nvidia_extra:
        failures.append("Requires-Dist has no nvidia extra")
    for anchor in EXPECTED_EXTRA_ANCHORS:
        if not any(anchor in entry for entry in nvidia_extra):
            failures.append(f"nvidia extra lacks the {anchor!r} anchor")

    groups: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in entry_points_raw.splitlines():
        if line.startswith("[") and line.endswith("]"):
            current = groups.setdefault(line[1:-1], [])
        elif line.strip() and current is not None:
            current.append(line.strip())
    for group, expected_key in EXPECTED_ENTRY_POINTS.items():
        entries = groups.get(group, [])
        if not any(entry.split("=", 1)[0].strip() == expected_key for entry in entries):
            failures.append(
                f"entry-point group {group!r} lacks the {expected_key!r} entry: {entries}"
            )

    if "sglang_fl/nvidia_runtime.py" not in names:
        failures.append("wheel does not contain sglang_fl/nvidia_runtime.py")

    test_payload = sorted(n for n in names if n.startswith("tests/") or "/tests/" in n)
    if test_payload:
        failures.append(f"wheel ships test files, e.g. {test_payload[:3]}")

    if failures:
        for failure in failures:
            print(f"FATAL: {failure}", file=sys.stderr)
        return 1

    print(f"[wheel] {name}=={version}")
    print(
        f"[wheel] nvidia extra: {len(nvidia_extra)} requirements "
        f"(anchors verified: {', '.join(EXPECTED_EXTRA_ANCHORS)})"
    )
    print(f"[wheel] entry points: {sorted(EXPECTED_ENTRY_POINTS)} verified")
    print("[wheel] contains sglang_fl/nvidia_runtime.py; no tests shipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
