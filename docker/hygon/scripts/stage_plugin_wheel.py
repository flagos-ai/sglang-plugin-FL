#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Stage the sglang_fl wheel for the Hygon two-wheel image.

Two modes:

* ``built-from-checkout`` (development candidate): ``python -m build --wheel``
  on the checkout. The zip timestamps come from the commit
  (``SOURCE_DATE_EPOCH`` = commit time), not from the clock, so two builds of
  one commit are byte-identical for a given build backend. This does not
  replace a published artifact: build_two_wheel.sh refuses to push such an
  image.
* ``published`` (release input): an already published, immutable wheel given
  by path and pinned by its SHA256. Nothing is built here.

Either way a provenance JSON is written next to the staged wheel:
distribution, version, mode, sha256 and ``source_commit`` - the commit the
wheel was built from. The post-build acceptance applies the same lineage rule
to it as to the derived wheels (docker/hygon/scripts/provenance_lineage.py).

Usage:
  stage_plugin_wheel.py --dest DIR --mode built-from-checkout --repo-root DIR
  stage_plugin_wheel.py --dest DIR --mode published --wheel FILE --sha256 HEX

Prints ``plugin-wheel: <name> sha256=<hex> mode=<mode>``; any violation
prints FATAL and exits 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

WHEEL_NAME = "sglang_fl-0.1.0-py3-none-any.whl"
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
BUILT, PUBLISHED = "built-from-checkout", "published"


class StageError(Exception):
    """A staging violation; the message is printed after ``FATAL:``."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments], capture_output=True, text=True, check=True
    ).stdout.strip()


def wheel_identity(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as archive:
        metadata_name = next(
            n for n in archive.namelist() if n.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
    return metadata["Name"], metadata["Version"]


def write_provenance(wheel: Path, record: dict) -> Path:
    provenance = wheel.with_suffix(".provenance.json")
    name, version = wheel_identity(wheel)
    document = {"distribution": name, "version": version, **record}
    provenance.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return provenance


def build_from_checkout(repo_root: Path, dest: Path) -> Path:
    """Build the development-candidate wheel from the checkout."""
    commit = git(repo_root, "rev-parse", "HEAD")
    epoch = git(repo_root, "log", "-1", "--format=%ct", "HEAD")
    dirty = bool(git(repo_root, "status", "--porcelain"))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(dest),
            str(repo_root),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        env={**os.environ, "SOURCE_DATE_EPOCH": epoch},
    )
    wheel = dest / WHEEL_NAME
    if not wheel.is_file():
        raise StageError(f"the build produced no {WHEEL_NAME} under {dest}")
    write_provenance(
        wheel,
        {
            "mode": BUILT,
            "sha256": sha256_file(wheel),
            "source_commit": commit,
            "source_date_epoch": int(epoch),
            "source_tree": "dirty" if dirty else "clean",
        },
    )
    return wheel


def stage_published(wheel: Path, sha256: str, dest: Path) -> Path:
    """Copy a published wheel after verifying its pin and provenance."""
    if not FULL_SHA256.match(sha256):
        raise StageError(
            f"the published wheel's pin is not a SHA256 digest: {sha256!r}"
        )
    if not wheel.is_file() or wheel.stat().st_size == 0:
        raise StageError(f"{wheel} is not a non-empty file")
    if wheel.name != WHEEL_NAME:
        raise StageError(f"{wheel.name} is not the expected artifact {WHEEL_NAME}")
    actual = sha256_file(wheel)
    if actual != sha256:
        raise StageError(
            f"SHA256 mismatch for {wheel.name}: pinned {sha256}, got {actual} "
            "(a published wheel is immutable; different content needs a new version)"
        )
    published = wheel.with_suffix(".provenance.json")
    if not published.is_file():
        raise StageError(f"{published} (provenance of the published wheel) is missing")
    record = json.loads(published.read_text(encoding="utf-8"))
    commit = record.get("source_commit")
    if not isinstance(commit, str) or not FULL_SHA.match(commit):
        raise StageError(f"{published} carries no full source_commit: {commit!r}")
    if record.get("sha256", sha256) != sha256:
        raise StageError(f"{published} describes another wheel: {record['sha256']}")
    dest.mkdir(parents=True, exist_ok=True)
    staged = dest / WHEEL_NAME
    staged.write_bytes(wheel.read_bytes())
    write_provenance(
        staged,
        {"mode": PUBLISHED, "sha256": sha256, "source_commit": commit},
    )
    return staged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mode", choices=[BUILT, PUBLISHED])
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    try:
        if args.mode == BUILT:
            if not args.repo_root:
                raise StageError("--mode built-from-checkout needs --repo-root")
            wheel = build_from_checkout(args.repo_root, args.dest)
        else:
            if not args.wheel or not args.sha256:
                raise StageError("--mode published needs --wheel and --sha256")
            wheel = stage_published(args.wheel, args.sha256, args.dest)
        name, version = wheel_identity(wheel)
    except (StageError, subprocess.CalledProcessError) as failure:
        print(f"FATAL: {failure}", file=sys.stderr)
        return 1
    print(
        f"plugin-wheel: {name}=={version} sha256={sha256_file(wheel)} mode={args.mode}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
