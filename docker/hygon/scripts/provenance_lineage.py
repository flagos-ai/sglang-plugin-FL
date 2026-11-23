#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Check that a wheel's provenance belongs to the history of an image.

The derived wheels are immutable artifacts bound by their SHA256 (the OCI
labels and the per-file manifests checked by verify_image.sh). They may have
been derived - or published - at an EARLIER commit than the image installing
them: staging-time derivation records the current HEAD, the pre-derived path
(SGLANG_DERIVED_WHEEL / XGRAMMAR_DERIVED_WHEEL, later a published wheel)
records the commit the artifact was produced at. Equality with the image's
source commit is therefore not the contract; lineage is: the derivation
commit must exist in this repository and be an ancestor of (or equal to)
the image's source-commit.

Usage: provenance_lineage.py <provenance.json> <image source-commit> [--repo DIR]
Prints FATAL and exits 1 on any violation.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def git(repo: Path, *arguments: str) -> int:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments], capture_output=True, text=True
    ).returncode


def lineage_violation(provenance: Path, image_commit: str, repo: Path) -> str | None:
    """Return why the provenance does not belong to the image, else None."""
    try:
        document = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, ValueError) as failure:
        return f"provenance {provenance} cannot be read: {failure}"
    derived_from = document.get("source_commit") if isinstance(document, dict) else None
    if not isinstance(derived_from, str) or not FULL_SHA.match(derived_from):
        return f"provenance {provenance.name} carries no full source_commit: {derived_from!r}"
    if not FULL_SHA.match(image_commit):
        return f"image source-commit is not a full commit id: {image_commit!r}"
    if git(repo, "cat-file", "-e", f"{derived_from}^{{commit}}") != 0:
        return (
            f"provenance was derived at {derived_from}, which is not a commit of {repo}"
        )
    if git(repo, "merge-base", "--is-ancestor", derived_from, image_commit) != 0:
        return (
            f"provenance was derived at {derived_from}, which is not an ancestor "
            f"of the image source-commit {image_commit}"
        )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provenance", type=Path)
    parser.add_argument("image_commit")
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    violation = lineage_violation(args.provenance, args.image_commit, args.repo)
    if violation:
        print(f"FATAL: {violation}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
