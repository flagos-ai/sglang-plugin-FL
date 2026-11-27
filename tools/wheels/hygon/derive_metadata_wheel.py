#!/usr/bin/env python3
"""Derive a metadata-corrected copy of a dependency wheel for the Hygon HCU image.

A dependency wheel may declare ``triton`` in its METADATA for Linux/x86_64.
The Hygon operations Runtime provides the Triton compiler as the
``/opt/flagtree`` module tree, NOT as a pip distribution, so a resolver that
honours that declaration pulls a second, standalone ``triton`` into the image.
This tool produces a NEW local-version copy of such a wheel with exactly that
declaration removed, so the image can install it without ``--excludes`` and
without a pip-check exemption (PR #180 r4627512386: "We can simply drop the
``Requires-Dist`` line from the built wheel").

Only METADATA is edited. Every payload byte, the member order (directory
entries included) and every member's zip attributes are carried over from the
pinned input wheel.

Contract (see tools/wheels/hygon/README.md):
  1. the input wheel must exist and hash to the pinned SHA256;
  2. filename, METADATA Name and METADATA Version must agree;
  3. the derived version must be ``<input version>+flagos.hcu<N>``: a new
     local version - the original version is never overwritten or
     impersonated - and its public part still satisfies an exact
     ``==<input version>`` requirement (PEP 440 ignores a candidate's local
     segment when the specifier has none);
  4. every ``--drop-requires-dist`` line must be present exactly once in the
     input METADATA, must name ``--drop-distribution`` and its marker must be
     ACTIVE on the HCU target: that is the declaration which makes the
     resolver install the standalone distribution there;
  5. the wheel is unpacked to a mktemp directory OUTSIDE the repository;
  6. METADATA surgery keeps every other byte: the ``Version`` line is
     rewritten and the dropped lines are deleted;
  7. the ``*.dist-info`` directory is renamed to the derived version and
     RECORD is rebuilt as standard CSV (``csv.writer``, ``\\n`` terminator);
  8. the repack is deterministic: input member order, per-member
     date/mode/compression kind preserved - two independent derivations are
     byte-identical;
  9. the finished wheel is re-verified through the strict gates
     (verify_metadata_wheel) BEFORE the script reports success; a failed gate
     deletes the output;
 10. a provenance JSON with a full per-file manifest (hashes read back from
     the FINISHED wheel) is written next to the output wheel;
 11. intermediate directories are removed on exit.

Usage:
  derive_metadata_wheel.py --input-wheel W --expected-input-sha256 H \
      --drop-requires-dist 'triton; platform_system == "Linux" and ...' \
      --derived-version V --output-dir D --source-commit SHA
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

from derive_sglang_wheel import rebuild_record, repo_root
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from verify_derived_sglang_wheel import (
    GateFailure,
    parse_wheel_filename,
    sha256_file,
    wheel_dist_info,
)
from verify_metadata_wheel import (
    LOCAL_SEGMENT_RE,
    target_environment,
    verify,
)

# The correction is made for exactly this target. A requirement may only be
# dropped when its marker is true here, i.e. when it is the declaration that
# makes the resolver install the standalone distribution on the HCU image.
HCU_TARGET = target_environment()


class DeriveFailure(Exception):
    """A derive contract step failed."""


def validate_drop_lines(drop_lines: list[str], drop_distribution: str) -> list:
    """Each dropped line names the drop distribution and is active on HCU."""
    target = canonicalize_name(drop_distribution)
    requirements = []
    for line in drop_lines:
        try:
            requirement = Requirement(line)
        except Exception as error:
            raise DeriveFailure(
                f"--drop-requires-dist is not a valid requirement: {line!r} ({error})"
            ) from error
        if canonicalize_name(requirement.name) != target:
            raise DeriveFailure(
                f"--drop-requires-dist names {requirement.name!r}, expected "
                f"{drop_distribution!r}: {line!r}"
            )
        active = (
            requirement.marker.evaluate(HCU_TARGET)
            if requirement.marker is not None
            else True
        )
        if active is not True:
            raise DeriveFailure(
                f"refusing to drop an inactive requirement on the HCU target "
                f"(only the declaration that is active there may be removed): {line!r}"
            )
        requirements.append(requirement)
    return requirements


def rewrite_metadata(
    metadata: bytes, old_version: str, new_version: str, drop_lines: list[str]
) -> bytes:
    """Rewrite the ``Version`` line and delete the dropped ``Requires-Dist``
    lines (each with its line terminator). Operates on the raw bytes of the
    header only, so every other byte - the body included - stays as it is."""
    old_line = f"Version: {old_version}".encode()
    new_line = f"Version: {new_version}".encode()
    drop = {f"Requires-Dist: {line}".encode() for line in drop_lines}
    version_hits, drop_hits = 0, {line: 0 for line in drop}
    out = []
    in_header = True
    for raw in metadata.splitlines(keepends=True):
        body = raw.rstrip(b"\r\n")
        terminator = raw[len(body) :]
        if in_header and body == b"":
            in_header = False
        if in_header and body == old_line:
            out.append(new_line + terminator)
            version_hits += 1
        elif in_header and body in drop:
            drop_hits[body] += 1
        else:
            out.append(raw)
    if version_hits != 1:
        raise DeriveFailure(
            f"expected exactly one {old_line!r} line in METADATA, found {version_hits}"
        )
    for line, hits in drop_hits.items():
        if hits != 1:
            raise DeriveFailure(
                f"expected exactly one {line!r} line in METADATA, found {hits}"
            )
    return b"".join(out)


def rename_member(name: str, old_dist: str, new_dist: str) -> str:
    if name == old_dist or name.startswith(old_dist + "/"):
        return new_dist + name[len(old_dist) :]
    return name


def repack(
    out_path: Path, src_zip: Path, root: Path, old_dist: str, new_dist: str
) -> None:
    """Deterministic repack from the INPUT archive's member list: order,
    directory entries, per-member date/mode/compression kind preserved;
    renamed dist-info members inherit the original members' attributes."""
    with zipfile.ZipFile(src_zip) as zin, zipfile.ZipFile(out_path, "w") as zout:
        for info in zin.infolist():
            if info.comment or info.extra:
                raise DeriveFailure(
                    "unsupported input wheel: member carries a zip comment or "
                    f"extra field ({info.filename})"
                )
            out_name = rename_member(info.filename, old_dist, new_dist)
            out_info = zipfile.ZipInfo(out_name, date_time=info.date_time)
            out_info.external_attr = info.external_attr
            out_info.create_system = info.create_system
            out_info.compress_type = info.compress_type
            if info.is_dir():
                zout.writestr(out_info, b"")
                continue
            data = (root / out_name).read_bytes()
            if info.compress_type == zipfile.ZIP_DEFLATED:
                zout.writestr(
                    out_info,
                    data,
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
            else:
                zout.writestr(out_info, data, compress_type=info.compress_type)


def derive(
    input_wheel: Path,
    expected_input_sha256: str,
    drop_lines: list[str],
    drop_distribution: str,
    derived_version: str,
    output_dir: Path,
    source_commit: str,
) -> dict:
    # git-free tool, but every temp path must still be absolute before any
    # tempdir exists
    input_wheel = input_wheel.resolve()
    # --- 1. input identity ------------------------------------------------
    if not input_wheel.is_file():
        raise DeriveFailure(f"input wheel not found: {input_wheel}")
    actual = sha256_file(input_wheel)
    if actual != expected_input_sha256:
        raise DeriveFailure(
            f"input wheel SHA256 {actual} != pinned {expected_input_sha256}"
        )

    # --- 2. filename / METADATA agreement ---------------------------------
    try:
        in_name, in_version, in_tags = parse_wheel_filename(input_wheel.name)
    except GateFailure as error:
        raise DeriveFailure(f"cannot parse input wheel filename: {error}") from error
    with zipfile.ZipFile(input_wheel) as zin:
        in_dist = wheel_dist_info(zin)
        in_metadata = zin.read(f"{in_dist}/METADATA")
    in_meta = BytesParser().parsebytes(in_metadata)
    if canonicalize_name(in_meta.get("Name") or "") != canonicalize_name(in_name):
        raise DeriveFailure(
            f"input filename names {in_name!r} but METADATA names "
            f"{in_meta.get('Name')!r}"
        )
    if in_meta.get("Version") != in_version:
        raise DeriveFailure(
            f"input filename version {in_version!r} != METADATA "
            f"{in_meta.get('Version')!r}"
        )
    dist_prefix = in_dist[: -len(".dist-info")]
    if not dist_prefix.endswith(in_version) or dist_prefix[: -len(in_version)] != (
        f"{in_name}-"
    ):
        raise DeriveFailure(
            f"dist-info directory {in_dist!r} does not name {in_name!r} {in_version!r}"
        )

    # --- 3. derived version: a NEW local version of the same public one ----
    local = LOCAL_SEGMENT_RE.search(derived_version)
    if local is None or derived_version[: local.start()] != in_version:
        raise DeriveFailure(
            f"derived version must be exactly {{input version}}+flagos.hcu<N> "
            f"(input {in_version!r}), got {derived_version!r}"
        )

    # --- 4. the dropped declarations --------------------------------------
    validate_drop_lines(drop_lines, drop_distribution)
    target = canonicalize_name(drop_distribution)
    remaining = [
        line
        for line in in_meta.get_all("Requires-Dist") or ()
        if line not in drop_lines
        and canonicalize_name(Requirement(line).name) == target
    ]

    new_dist = f"{in_name}-{derived_version}.dist-info"
    out_name = f"{in_name}-{derived_version}-{in_tags}.whl"

    # --- 5/6/7/8. unpack, surgery, RECORD, deterministic repack -----------
    root_guard = repo_root()
    workdir = Path(tempfile.mkdtemp(prefix="hcu-metadata-"))
    try:
        if root_guard is not None:
            import os

            guard = str(root_guard.resolve())
            if str(workdir.resolve()).startswith(guard + os.sep):
                raise DeriveFailure("mktemp directory landed inside the repository")
        unpack_root = workdir / "unpacked"
        with zipfile.ZipFile(input_wheel) as zin:
            zin.extractall(unpack_root)

        metadata_path = unpack_root / in_dist / "METADATA"
        metadata_path.write_bytes(
            rewrite_metadata(in_metadata, in_version, derived_version, drop_lines)
        )
        (unpack_root / in_dist).rename(unpack_root / new_dist)
        (unpack_root / new_dist / "RECORD").write_bytes(
            rebuild_record(unpack_root, new_dist)
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / out_name
        repack(out_path, input_wheel, unpack_root, in_dist, new_dist)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    # --- 9. strict gates ----------------------------------------------------
    try:
        report = verify(
            input_wheel,
            out_path,
            drop_lines,
            drop_distribution,
            expected_input_sha256,
            HCU_TARGET,
        )
    except GateFailure as failure:
        out_path.unlink(missing_ok=True)
        raise DeriveFailure(
            f"derived wheel rejected by the strict gates: {failure}"
        ) from failure

    # --- 10. provenance with a FULL per-file manifest, read back from the
    # FINISHED wheel: the image acceptance compares every installed file
    # against it, binding the image to the exact artifact content
    with zipfile.ZipFile(out_path) as zout:
        bound_files = {
            info.filename: hashlib.sha256(zout.read(info.filename)).hexdigest()
            for info in zout.infolist()
            if not info.is_dir()
        }
    provenance = {
        "input_wheel": input_wheel.name,
        "input_sha256": expected_input_sha256,
        "input_version": in_version,
        "drop_distribution": canonicalize_name(drop_distribution),
        "dropped_requires_dist": sorted(drop_lines),
        "remaining_requires_dist": remaining,
        "target_environment": HCU_TARGET,
        "derived_wheel": out_name,
        "derived_version": derived_version,
        "derived_sha256": report["derived_sha256"],
        "bound_files_manifest": bound_files,
        "source_commit": source_commit,
        "deterministic_packaging": (
            "input member order incl. directory entries; per-member "
            "date/mode/compression kind preserved (STORED members stay "
            "STORED); DEFLATE level 9; sorted CSV RECORD"
        ),
    }
    provenance_path = out_path.with_suffix(".provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a metadata-corrected local-version copy of a dependency "
            "wheel for the Hygon HCU image"
        )
    )
    parser.add_argument("--input-wheel", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument(
        "--drop-requires-dist",
        action="append",
        required=True,
        metavar="LINE",
        help="exact Requires-Dist value to delete (repeatable)",
    )
    parser.add_argument(
        "--drop-distribution",
        required=True,
        help="the distribution every dropped line must name (e.g. triton)",
    )
    parser.add_argument("--derived-version", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    try:
        provenance = derive(
            args.input_wheel,
            args.expected_input_sha256,
            args.drop_requires_dist,
            args.drop_distribution,
            args.derived_version,
            args.output_dir,
            args.source_commit,
        )
    except DeriveFailure as failure:
        print(f"DERIVE-FAILURE: {failure}", file=sys.stderr)
        return 1
    print("DERIVED-WHEEL::" + json.dumps(provenance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
