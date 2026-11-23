#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Write the publication manifest for the Hygon platform wheels.

Reviewers require the wheels a release image installs to come from
resource.flagos.net (PR #180 r4627443492). Until the artifact owner
publishes them, staging produces LOCAL DEVELOPMENT CANDIDATES; this tool
writes the manifest that a publication would consume. Per artifact it
records the distribution, version, filename, size, SHA256, the suggested
Nexus repository and path, and - for a derived wheel - the provenance that
ties it to its pinned input.

The manifest is disposable build evidence: never committed, never in the
image. A release build later consumes the PUBLISHED wheels by pinning the
same SHA256 values (docker/hygon/stage_wheels.sh verifies a pre-derived
wheel against the pinned digest before it is staged).

Usage: publication_manifest.py --context DIR [--out PATH]
  DIR is the staging context assembled by docker/hygon/stage_wheels.sh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

NEXUS = "https://resource.flagos.net/repository"
REPOSITORY = "flagos-pypi-hygon"
# filename stem -> (provenance JSON next to it, provenance kind)
DERIVED_PROVENANCE = {
    "sglang-0.5.18+flagos.hcu1-cp310-cp310-linux_x86_64": "derived",
    "xgrammar-0.2.1+flagos.hcu1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64": (
        "metadata-corrected"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def wheel_identity(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as archive:
        metadata_name = next(
            n for n in archive.namelist() if n.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
    return metadata["Name"], metadata["Version"]


def describe(path: Path) -> dict:
    name, version = wheel_identity(path)
    entry = {
        "distribution": name,
        "version": version,
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "status": "LOCAL DEVELOPMENT CANDIDATE - not a published wheel",
        "nexus": {
            "repository": REPOSITORY,
            "path": f"packages/{name}/{version}/{path.name}",
            "download_url": f"{NEXUS}/{REPOSITORY}/packages/{name}/{version}/{path.name}",
            "simple_index": f"{NEXUS}/{REPOSITORY}/simple/{name}/",
        },
    }
    kind = DERIVED_PROVENANCE.get(path.name.removesuffix(".whl"))
    if kind:
        provenance_path = path.with_suffix(".provenance.json")
        if provenance_path.is_file():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            keys = (
                "input_wheel",
                "input_sha256",
                "patch",
                "patch_sha256",
                "dropped_requires_dist",
                "source_commit",
            )
            entry["derived"] = {
                "kind": kind,
                **{key: provenance[key] for key in keys if key in provenance},
                "provenance_file": provenance_path.name,
            }
        return entry
    # the plugin wheel: staged by docker/hygon/scripts/stage_plugin_wheel.py
    # either as a development candidate or as the published artifact
    provenance_path = path.with_suffix(".provenance.json")
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("mode") == "published":
            entry["status"] = (
                "PUBLISHED sglang_fl ARTIFACT - consumed by its SHA256 pin "
                "(SGLANG_FL_WHEEL)"
            )
            entry["built_from"] = (
                "the published wheel (SGLANG_FL_WHEEL); nothing is built here"
            )
        else:
            entry["built_from"] = (
                "this repository checkout (python -m build --wheel; "
                "SOURCE_DATE_EPOCH = commit time)"
            )
        entry["provenance"] = {
            key: provenance[key]
            for key in ("mode", "source_commit", "sha256")
            if key in provenance
        }
    else:
        entry["built_from"] = "this repository checkout (python -m build --wheel)"
    return entry


def build(context: Path) -> dict:
    artifacts = context / "artifacts"
    wheels = sorted(artifacts.glob("*.whl"))
    assert wheels, f"no wheels under {artifacts}"
    return {
        "schema": "flagos-hygon-wheel-publication-manifest/1",
        "target": f"{NEXUS}/{REPOSITORY}",
        "status": (
            "LOCAL DEVELOPMENT CANDIDATES. Nothing here is published; a "
            "release image consumes these wheels only after the artifact "
            "owner publishes them at the suggested paths and staging "
            "re-verifies the SHA256 values below."
        ),
        "policy": [
            "never overwrite or republish an existing version",
            "one name/version pair carries exactly one content: after "
            "publication any content change needs the next local version "
            "(hcu2)",
            "a published wheel is immutable; release builds pin its SHA256",
        ],
        "artifacts": [describe(wheel) for wheel in wheels],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    manifest = build(args.context)
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    out = args.out or args.context / "publication-manifest.json"
    out.write_text(text, encoding="utf-8")
    for artifact in manifest["artifacts"]:
        print(
            f"{artifact['distribution']}=={artifact['version']}  "
            f"sha256={artifact['sha256']}  -> {artifact['nexus']['path']}"
        )
    print(f"publication manifest (disposable): {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, StopIteration) as failure:
        print(f"FATAL: {failure}", file=sys.stderr)
        raise SystemExit(1) from failure
