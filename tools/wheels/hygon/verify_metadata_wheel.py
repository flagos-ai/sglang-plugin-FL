#!/usr/bin/env python3
"""Strict diff gates for a metadata-only derived dependency wheel.

Compares the derived wheel against the SHA-pinned original wheel and enforces
that NOTHING outside the reviewed allowlist differs. The verifier is
deliberately independent of the derive script: it never calls the derive
script's transformation, it rebuilds its expectations from the ORIGINAL
wheel and checks the metadata twice - parsed (email parser + Requirement)
and on the raw bytes.

  allowed to change
    - the METADATA ``Version`` line (input -> derived local version);
    - the listed ``Requires-Dist`` lines, which must be DELETED - nothing
      else may be added, removed or reordered;
    - the ``*.dist-info`` directory name;
    - RECORD (rebuilt as standard CSV; every hash re-validated here).

  must stay identical
    - every payload file, byte for byte (native payload reported separately);
    - the member list and its order, directory entries included;
    - every member's zip date, mode, create system and compression kind
      (STORED members stay STORED);
    - WHEEL (tags), entry_points, licenses, every other dist-info file;
    - the wheel tags (python/abi/platform).

Additional structural proofs
    - RECORD is standard CSV: parsed with ``csv.reader``, exactly three
      fields per row (comma-in-path rows are quoted), LF-terminated, and
      byte-identical to its own rows re-serialised through ``csv.writer``
      with ``lineterminator="\\n"``;
    - filename / METADATA / dist-info directory / derived version agree, and
      the derived version is ``<input version>+flagos.hcu<N>`` (the original
      version is never reused);
    - every dropped line is present exactly once in the original METADATA,
      names the drop distribution and is ACTIVE in the target environment.

Exit code 0 means every gate passed. ``--json PATH`` additionally writes the
machine-readable report (disposable evidence; never committed).

Usage:
  verify_metadata_wheel.py --input-wheel W --expected-input-sha256 H \
      --derived-wheel W --drop-requires-dist LINE --drop-distribution NAME \
      [--expected-derived-sha256 H] [--json PATH]
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from verify_derived_sglang_wheel import (
    NATIVE_SUFFIXES,
    GateFailure,
    parse_wheel_filename,
    sha256_file,
    wheel_dist_info,
)

LOCAL_SEGMENT_RE = re.compile(r"\+flagos\.hcu[1-9][0-9]*$")
DEFAULT_TARGET_PYTHON = "3.10.20"


def target_environment(python_full_version: str = DEFAULT_TARGET_PYTHON) -> dict:
    """The HCU image target the correction is made for (the operations
    Runtime's interpreter and platform)."""
    return {
        "implementation_name": "cpython",
        "implementation_version": python_full_version,
        "os_name": "posix",
        "platform_machine": "x86_64",
        "platform_release": "",
        "platform_system": "Linux",
        "platform_version": "",
        "platform_python_implementation": "CPython",
        "python_full_version": python_full_version,
        "python_version": ".".join(python_full_version.split(".")[:2]),
        "sys_platform": "linux",
    }


def rename_member(name: str, old_dist: str, new_dist: str) -> str:
    if name == old_dist or name.startswith(old_dist + "/"):
        return new_dist + name[len(old_dist) :]
    return name


def read_record(record: bytes) -> list[list[str]]:
    """RECORD rows through ``csv.reader``; standard CSV only.

    Every row must parse to exactly three fields, the file must be
    LF-terminated, and it must equal its own rows re-serialised through
    ``csv.writer(lineterminator="\\n")`` - a RECORD produced by any other
    means (unquoted comma paths, CR terminators, stray quoting) is rejected.
    """
    if b"\r" in record:
        raise GateFailure("RECORD carries CR line endings (records are LF-terminated)")
    rows = list(csv.reader(io.StringIO(record.decode("utf-8"), newline="")))
    for row in rows:
        if not row:
            raise GateFailure("RECORD carries an empty row")
        if len(row) != 3:
            raise GateFailure(f"RECORD row is not exactly three CSV fields: {row!r}")
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    if buffer.getvalue().encode("utf-8") != record:
        raise GateFailure(
            "RECORD is not the canonical csv.writer(lineterminator='\\n') "
            "serialisation of its own rows"
        )
    return rows


def urlsafe_sha256(data: bytes) -> str:
    return (
        base64.urlsafe_b64encode(hashlib.sha256(data).digest())
        .rstrip(b"=")
        .decode("ascii")
    )


def verify(
    input_wheel: Path,
    derived_wheel: Path,
    drop_lines: list[str],
    drop_distribution: str,
    expected_input_sha256: str,
    target: dict | None = None,
    expected_derived_sha256: str | None = None,
) -> dict:
    """Run every gate; raise GateFailure on the first violation."""
    target = target or target_environment()
    drop_lines = list(drop_lines)
    drop_name = canonicalize_name(drop_distribution)

    # git-free verifier, but the wheels are re-read several times: absolute
    input_wheel, derived_wheel = input_wheel.resolve(), derived_wheel.resolve()

    # --- V0: artifact identity -------------------------------------------
    if sha256_file(input_wheel) != expected_input_sha256:
        raise GateFailure("input wheel SHA256 does not match the pinned value")
    derived_sha256 = sha256_file(derived_wheel)
    if expected_derived_sha256 and derived_sha256 != expected_derived_sha256:
        raise GateFailure(
            f"derived wheel SHA256 {derived_sha256} != expected "
            f"{expected_derived_sha256}"
        )

    with zipfile.ZipFile(input_wheel) as zin, zipfile.ZipFile(derived_wheel) as zout:
        in_infos, out_infos = zin.infolist(), zout.infolist()
        in_names, out_names = (
            [i.filename for i in in_infos],
            [i.filename for i in out_infos],
        )
        if len(set(in_names)) != len(in_names) or len(set(out_names)) != len(out_names):
            raise GateFailure("a wheel carries duplicate member names")
        in_dist, out_dist = wheel_dist_info(zin), wheel_dist_info(zout)

        # --- V1: the ORIGINAL wheel is self-consistent -------------------
        in_name, in_version, in_tags = parse_wheel_filename(input_wheel.name)
        in_metadata = zin.read(f"{in_dist}/METADATA")
        in_meta = BytesParser().parsebytes(in_metadata)
        if canonicalize_name(in_meta.get("Name") or "") != canonicalize_name(in_name):
            raise GateFailure(
                f"input wheel: filename names {in_name!r}, METADATA names "
                f"{in_meta.get('Name')!r}"
            )
        if in_meta.get("Version") != in_version:
            raise GateFailure(
                f"input wheel: filename version {in_version!r} != METADATA "
                f"{in_meta.get('Version')!r}"
            )
        if in_dist != f"{in_name}-{in_version}.dist-info":
            raise GateFailure(
                f"input wheel: dist-info directory {in_dist!r} does not match "
                f"its filename {input_wheel.name!r}"
            )

        # --- V2: filename / dist-info / version consistency --------------
        out_name, out_version, out_tags = parse_wheel_filename(derived_wheel.name)
        local = LOCAL_SEGMENT_RE.search(out_version)
        if local is None or out_version[: local.start()] != in_version:
            raise GateFailure(
                f"derived version {out_version!r} is not "
                f"{{input version}}+flagos.hcu<N> with input {in_version!r}: the "
                "original version must never be overwritten or impersonated"
            )
        if out_name != in_name:
            raise GateFailure(f"derived wheel name {out_name!r} != {in_name!r}")
        if out_tags != in_tags:
            raise GateFailure(f"wheel tags drifted: {in_tags!r} -> {out_tags!r}")
        expected_dist = f"{in_name}-{out_version}.dist-info"
        if out_dist != expected_dist:
            raise GateFailure(
                f"dist-info directory {out_dist!r} != expected {expected_dist!r}"
            )
        out_metadata = zout.read(f"{out_dist}/METADATA")
        out_meta = BytesParser().parsebytes(out_metadata)
        if out_meta.get("Version") != out_version:
            raise GateFailure(
                f"derived filename version {out_version!r} != METADATA "
                f"{out_meta.get('Version')!r}"
            )
        if canonicalize_name(out_meta.get("Name") or "") != canonicalize_name(in_name):
            raise GateFailure(
                f"derived METADATA names {out_meta.get('Name')!r}, expected {in_name!r}"
            )

        # --- V3: METADATA - parsed view ----------------------------------
        if not drop_lines:
            raise GateFailure("no Requires-Dist line was declared as dropped")
        in_requires = list(in_meta.get_all("Requires-Dist") or ())
        out_requires = list(out_meta.get_all("Requires-Dist") or ())
        for line in drop_lines:
            if in_requires.count(line) != 1:
                raise GateFailure(
                    f"dropped line is not present exactly once in the original "
                    f"METADATA: {line!r}"
                )
            requirement = Requirement(line)
            if canonicalize_name(requirement.name) != drop_name:
                raise GateFailure(
                    f"dropped line names {requirement.name!r}, not {drop_name!r}"
                )
            active = (
                requirement.marker.evaluate(target)
                if requirement.marker is not None
                else True
            )
            if active is not True:
                raise GateFailure(
                    f"dropped line is not active in the target environment "
                    f"(it is not the declaration that pulls the standalone "
                    f"distribution in): {line!r}"
                )
        expected_requires = [line for line in in_requires if line not in drop_lines]
        if out_requires != expected_requires:
            raise GateFailure(
                "Requires-Dist differs beyond the dropped lines: removed "
                f"{sorted(set(in_requires) - set(out_requires))}, added "
                f"{sorted(set(out_requires) - set(in_requires))}, or reordered"
            )
        drifted = sorted(
            key
            for key in set(in_meta) | set(out_meta)
            if key not in ("Version", "Requires-Dist")
            and in_meta.get_all(key) != out_meta.get_all(key)
        )
        if drifted:
            raise GateFailure(f"METADATA headers drifted: {drifted}")

        # --- V3: METADATA - raw-byte view --------------------------------
        # the derived file must be the original with the single Version line
        # rewritten and the dropped lines deleted, byte for byte
        in_lines = in_metadata.splitlines(keepends=True)
        version_idx = [
            i for i, line in enumerate(in_lines) if line.startswith(b"Version:")
        ]
        if len(version_idx) != 1:
            raise GateFailure("original METADATA has an unexpected Version-line count")
        drop_bodies = {f"Requires-Dist: {line}".encode() for line in drop_lines}
        rebuilt = []
        dropped = 0
        in_header = True
        for i, line in enumerate(in_lines):
            body = line.rstrip(b"\r\n")
            if in_header and body == b"":
                in_header = False  # the description body is never touched
            if in_header and body in drop_bodies:
                dropped += 1
                continue
            if i == version_idx[0]:
                line = line.replace(
                    f"Version: {in_version}".encode(),
                    f"Version: {out_version}".encode(),
                    1,
                )
            rebuilt.append(line)
        if dropped != len(drop_lines):
            raise GateFailure("raw-byte view: dropped line count mismatch")
        if b"".join(rebuilt) != out_metadata:
            raise GateFailure(
                "METADATA changed beyond the Version line and the dropped lines"
            )

        # --- V4: member list and order -----------------------------------
        renamed = [rename_member(n, in_dist, out_dist) for n in in_names]
        if renamed != out_names:
            only_in = sorted(set(renamed) - set(out_names))
            only_out = sorted(set(out_names) - set(renamed))
            raise GateFailure(
                f"member list/order differs; only in original: {only_in[:5]}; "
                f"only in derived: {only_out[:5]}; "
                f"same set: {set(renamed) == set(out_names)}"
            )

        # --- V5/V6: per-member attributes and payload bytes --------------
        unchanged_differ, attr_differ, native_report = [], [], {}
        for info_in, info_out in zip(in_infos, out_infos):
            name = rename_member(info_in.filename, in_dist, out_dist)
            if (
                info_in.compress_type != info_out.compress_type
                or info_in.date_time != info_out.date_time
                or info_in.external_attr != info_out.external_attr
                or info_in.create_system != info_out.create_system
            ):
                attr_differ.append(name)
            if info_in.comment or info_in.extra:
                raise GateFailure(
                    f"unsupported input wheel: member carries a zip comment or "
                    f"extra field ({info_in.filename})"
                )
            if name.endswith((f"{out_dist}/METADATA", f"{out_dist}/RECORD")):
                continue
            data_in, data_out = zin.read(info_in), zout.read(info_out)
            if data_in != data_out:
                unchanged_differ.append(name)
            if name.endswith(NATIVE_SUFFIXES):
                native_report[name] = data_in == data_out
        if attr_differ:
            raise GateFailure(f"zip attributes drifted: {attr_differ[:10]}")
        if unchanged_differ:
            raise GateFailure(
                f"files outside the allowlist drifted: {unchanged_differ[:10]}"
            )
        if not all(native_report.values()):
            raise GateFailure(
                f"native payload changed: {[k for k, ok in native_report.items() if not ok]}"
            )
        if f"{out_dist}/WHEEL" not in out_names:
            raise GateFailure("derived wheel lacks its WHEEL file")

        # --- V7: RECORD is standard CSV and valid ------------------------
        record_rel = f"{out_dist}/RECORD"
        rows = read_record(zout.read(record_rel))
        file_names = {n for n in out_names if not n.endswith("/")}
        if len(rows) != len(file_names):
            raise GateFailure(
                f"RECORD has {len(rows)} rows for {len(file_names)} file members"
            )
        seen = set()
        for path, digest, size in rows:
            if path in seen:
                raise GateFailure(f"RECORD lists {path!r} twice")
            seen.add(path)
            if path not in file_names:
                raise GateFailure(f"RECORD lists a non-member: {path!r}")
            if path == record_rel:
                if (digest, size) != ("", ""):
                    raise GateFailure("RECORD's own entry must carry no hash")
                continue
            data = zout.read(path)
            if digest != f"sha256={urlsafe_sha256(data)}" or size != str(len(data)):
                raise GateFailure(f"RECORD hash/size wrong for {path}")
        if seen != file_names:
            raise GateFailure("RECORD member set mismatch")

    return {
        "input_sha256": expected_input_sha256,
        "derived_sha256": derived_sha256,
        "input_wheel": input_wheel.name,
        "derived_wheel": derived_wheel.name,
        "input_version": in_version,
        "derived_version": out_version,
        "drop_distribution": drop_name,
        "dropped_requires_dist": sorted(drop_lines),
        "remaining_requires_dist": [
            line
            for line in out_requires
            if canonicalize_name(Requirement(line).name) == drop_name
        ],
        "member_list_and_order_identical": True,
        "members": len(in_names),
        "zip_attributes_identical": True,
        "payload_byte_identical": True,
        "native_payload": {
            "files": sorted(native_report),
            "byte_identical": True,
        },
        "metadata_headers_differ": [],
        "metadata_raw_bytes_checked": True,
        "wheel_tags_identical": True,
        "record_standard_csv_lf": True,
        "record_rows": len(rows),
        "target_environment": target,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strict diff gates for a metadata-only derived dependency wheel"
    )
    parser.add_argument("--input-wheel", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--derived-wheel", type=Path, required=True)
    parser.add_argument("--expected-derived-sha256", default=None)
    parser.add_argument("--drop-requires-dist", action="append", required=True)
    parser.add_argument("--drop-distribution", required=True)
    parser.add_argument(
        "--target-python",
        default=DEFAULT_TARGET_PYTHON,
        help="full python version of the HCU target image",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="write the machine-readable report here"
    )
    args = parser.parse_args()

    try:
        report = verify(
            args.input_wheel,
            args.derived_wheel,
            args.drop_requires_dist,
            args.drop_distribution,
            args.expected_input_sha256,
            target_environment(args.target_python),
            args.expected_derived_sha256,
        )
    except GateFailure as failure:
        print(f"GATE-FAILURE: {failure}", file=sys.stderr)
        return 1
    print("DERIVED-WHEEL-VERIFIED::" + json.dumps(report, sort_keys=True))
    if args.json:
        args.json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
