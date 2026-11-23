#!/usr/bin/env python3
"""Acceptance checks for the staged sglang-plugin-FL wheel (Hygon).

Verifies the built wheel carries the two-wheel Hygon contract before it
enters any image build context:

  - METADATA Name/Version match the project,
  - the ``hygon`` optional dependency group exists, carries the exact
    compatibility anchors (flagos SGLang pin, runtime FlagGems line and the
    reviewed datasets/fsspec compatibility pin), and contains exactly three
    requirements,
  - no Requires-Dist anywhere in the wheel names ``triton`` (the Runtime
    provides the Triton compiler as the /opt/flagtree module tree, not as
    a pip distribution),
  - both SGLang entry-point groups are registered,
  - the Hygon vendor attention backend and dispatch configuration ship in
    the wheel,
  - no test files ship in the wheel.

Usage: check_plugin_wheel.py <wheel> [--repo-root DIR]
"""

import argparse
import configparser
import re
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

EXPECTED_EXTRA_ANCHORS = (
    # the derived HCU wheel (tools/wheels/hygon/derive_sglang_wheel.py)
    "sglang==0.5.18+flagos.hcu1",
    "flag-gems==5.4.0",
    "datasets==5.2.0",
)
EXPECTED_EXTRA_COUNT = 3
EXCLUDED_DISTRIBUTION = "triton"
EXPECTED_ENTRY_POINTS = {
    "sglang.srt.platforms": ("sglang_fl", "sglang_fl:activate_platform"),
    "sglang.srt.plugins": ("sglang_fl", "sglang_fl:load_plugin"),
}
REQUIRED_PAYLOAD = (
    "sglang_fl/dispatch/backends/vendor/hygon/impl/attention_backend.py",
    "sglang_fl/dispatch/config/hygon.yaml",
)


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

    failures: list[str] = []
    with zipfile.ZipFile(args.wheel) as archive:
        names = archive.namelist()
        metadata = Parser().parsestr(
            archive.read("sglang_fl-0.1.0.dist-info/METADATA").decode("utf-8")
        )

        if metadata.get("Name") != "sglang_fl":
            failures.append(f"METADATA Name is {metadata.get('Name')!r}")
        expected_version = project_version(args.repo_root)
        if metadata.get("Version") != expected_version:
            failures.append(
                f"METADATA Version {metadata.get('Version')!r} != project "
                f"{expected_version!r}"
            )

        hygon_extra = [
            line.strip()
            for line in metadata.get_all("Requires-Dist", ())
            if 'extra == "hygon"' in line
        ]
        anchors = sorted(
            requirement.split(";")[0].strip() for requirement in hygon_extra
        )
        if len(hygon_extra) != EXPECTED_EXTRA_COUNT:
            failures.append(
                f"[hygon] extra has {len(hygon_extra)} requirements, "
                f"expected {EXPECTED_EXTRA_COUNT}: {hygon_extra}"
            )
        for anchor in EXPECTED_EXTRA_ANCHORS:
            if anchor not in anchors:
                failures.append(f"[hygon] extra lacks the anchor {anchor!r}")

        for requirement in metadata.get_all("Requires-Dist", ()):
            name = re.split(r"[\s\[<>=!~;]", requirement.strip(), maxsplit=1)[0]
            if name.lower() == EXCLUDED_DISTRIBUTION:
                failures.append(
                    f"wheel declares {EXCLUDED_DISTRIBUTION!r}: {requirement!r}"
                )

        entry_points_text = archive.read(
            "sglang_fl-0.1.0.dist-info/entry_points.txt"
        ).decode("utf-8")
        parser = configparser.ConfigParser()
        parser.read_string(entry_points_text)
        for group, (name, value) in EXPECTED_ENTRY_POINTS.items():
            if not parser.has_section(group) or not parser.has_option(group, name):
                failures.append(f"entry-point {group}: {name} missing")
            elif parser.get(group, name) != value:
                failures.append(
                    f"entry-point {group}: {name} -> "
                    f"{parser.get(group, name)!r}, expected {value!r}"
                )

        for payload in REQUIRED_PAYLOAD:
            if payload not in names:
                failures.append(f"required payload missing from wheel: {payload}")

        shipped_tests = [name for name in names if name.startswith(("tests/", "test_"))]
        if shipped_tests:
            failures.append(f"test files ship in the wheel: {shipped_tests[:5]}")

    if failures:
        print("FATAL: plugin wheel rejected:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(
        f"plugin wheel accepted: sglang_fl=={project_version(args.repo_root)} "
        f"with the [hygon] two-wheel contract"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
