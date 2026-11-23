#!/usr/bin/env python3
"""Derive the Hygon HCU SGLang wheel from the SHA-pinned FlagOS wheel.

Folds the reviewed HCU gating patch (``sglang-v0.5.18-hcu.patch``) into a
copy of the exact published ``sglang-0.5.18+flagos`` wheel and republishes
it under the NEW local version ``0.5.18+flagos.hcu1``, so the product image
installs a normal wheel and stops patching site-packages at build time.

Contract (see tools/wheels/hygon/README.md):
  1. the input wheel must exist and hash to the pinned SHA256;
  2. its distribution must be ``sglang``;
  3. its version must be exactly ``0.5.18+flagos``;
  4. the derived version must differ and be exactly ``0.5.18+flagos.hcu1``;
  5. the wheel is unpacked to a mktemp directory OUTSIDE the repository;
  6. the patch is checked then applied there (``git apply -p2``);
  7. only these changes are made:
       - METADATA ``Version`` line,
       - the ``*.dist-info`` directory name,
       - the wheel filename,
       - RECORD (rebuilt),
       - ``sglang/_version.py`` ONLY when it carried the input
         distribution version (the published wheel's copy says
         ``0.5.18``, so it stays byte-identical);
  8. the output zip is deterministic: input member order, per-member
     timestamps/modes/attributes and compression kinds are preserved, so
     two independent derivations are byte-identical;
  9. the finished wheel is re-verified through the strict diff gates
     (verify_derived_sglang_wheel) BEFORE the script reports success;
 10. a provenance JSON is written next to the output wheel;
 11. intermediate directories are removed on exit;
 12. any unexpected difference fails the run.

Usage:
  derive_sglang_wheel.py --input-wheel W --expected-input-sha256 H \
      --patch P --output-dir D --derived-version V --source-commit SHA
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_derived_sglang_wheel import (
    INPUT_DISTRIBUTION,
    INPUT_VERSION,
    GateFailure,
    collect_files,
    parse_patch_files,
    parse_wheel_filename,
    run_git_apply,
    sha256_file,
    verify,
    wheel_dist_info,
)

DERIVED_VERSION = "0.5.18+flagos.hcu1"


class DeriveFailure(Exception):
    """A derive contract step failed."""


def repo_root() -> Path:
    try:
        out = subprocess.run(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parent),
                "rev-parse",
                "--show-toplevel",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(out.stdout.strip())
    except (subprocess.CalledProcessError, OSError):
        return None


def rebuild_record(root: Path, dist_info: str) -> bytes:
    """RECORD lines for the final tree, sorted by path (stable order).

    Unlike the published wheel's RECORD - which still references the stale
    ``sglang-0.5.18.dist-info/`` prefix although its archive ships the
    ``+flagos`` directory - the rebuilt RECORD names the real new
    dist-info directory.
    """
    import base64

    lines = []
    for rel in sorted(collect_files(root)):
        if rel == f"{dist_info}/RECORD":
            lines.append(f"{rel},,")
            continue
        data = (root / rel).read_bytes()
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            .rstrip(b"=")
            .decode("ascii")
        )
        lines.append(f"{rel},sha256={digest},{len(data)}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def rewrite_metadata_version(metadata: bytes, old: str, new: str) -> bytes:
    old_line = f"Version: {old}".encode()
    new_line = f"Version: {new}".encode()
    lines = metadata.split(b"\n")
    hits = [i for i, line in enumerate(lines) if line == old_line]
    if len(hits) != 1:
        raise DeriveFailure(
            f"expected exactly one {old_line!r} line in METADATA, found {len(hits)}"
        )
    lines[hits[0]] = new_line
    return b"\n".join(lines)


def maybe_rewrite_package_version(root: Path, old: str, new: str) -> bool:
    """Rewrite ``sglang/_version.py`` only if it carried the distribution
    version. The published wheel's copy hardcodes ``0.5.18`` (no local
    segment), matching upstream's setuptools-scm output, so it is left
    byte-identical and ``sglang.__version__`` stays consistent with the
    original wheel."""
    path = root / "sglang" / "_version.py"
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    marker = f"__version__ = version = '{old}'"
    if marker not in text:
        return False
    path.write_text(
        text.replace(marker, f"__version__ = version = '{new}'"), encoding="utf-8"
    )
    return True


def build_wheel(
    out_path: Path, src_zip: Path, root: Path, old_dist: str, new_dist: str
) -> None:
    """Deterministic repack: input member order; per-member date, mode and
    compression kind preserved; renamed dist-info members inherit the
    original members' attributes."""
    with zipfile.ZipFile(src_zip) as zin, zipfile.ZipFile(out_path, "w") as zout:
        for info in zin.infolist():
            if info.is_dir():
                continue
            out_name = info.filename
            if out_name.startswith(old_dist + "/"):
                out_name = new_dist + "/" + out_name.split("/", 1)[1]
            data = (root / out_name).read_bytes()
            out_info = zipfile.ZipInfo(out_name, date_time=info.date_time)
            out_info.external_attr = info.external_attr
            out_info.create_system = info.create_system
            out_info.compress_type = zipfile.ZIP_DEFLATED
            zout.writestr(
                out_info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9
            )


def derive(
    input_wheel: Path,
    expected_input_sha256: str,
    patch: Path,
    output_dir: Path,
    derived_version: str,
    source_commit: str,
) -> dict:
    # git apply runs with the unpack directory as cwd: every path must be
    # absolute before the tempdirs exist
    input_wheel, patch = input_wheel.resolve(), patch.resolve()
    # --- 1. input identity ------------------------------------------------
    if not input_wheel.is_file():
        raise DeriveFailure(f"input wheel not found: {input_wheel}")
    actual = sha256_file(input_wheel)
    if actual != expected_input_sha256:
        raise DeriveFailure(
            f"input wheel SHA256 {actual} != pinned {expected_input_sha256}"
        )

    # --- 2/3. distribution and version gates ------------------------------
    try:
        _in_name, in_version, in_tags = parse_wheel_filename(input_wheel.name)
    except Exception as error:  # noqa: BLE001 - surfaced as contract failure
        raise DeriveFailure(f"cannot parse input wheel filename: {error}")
    with zipfile.ZipFile(input_wheel) as zin:
        in_dist = wheel_dist_info(zin)
        in_meta = BytesParser().parsebytes(zin.read(f"{in_dist}/METADATA"))
    if (in_meta.get("Name") or "").lower() != INPUT_DISTRIBUTION:
        raise DeriveFailure(
            f"input distribution is {in_meta.get('Name')!r}, expected "
            f"{INPUT_DISTRIBUTION!r}"
        )
    if in_meta.get("Version") != INPUT_VERSION:
        raise DeriveFailure(
            f"input version is {in_meta.get('Version')!r}, expected {INPUT_VERSION!r}"
        )
    if in_version != INPUT_VERSION:
        raise DeriveFailure(
            f"input filename version {in_version!r} != {INPUT_VERSION!r}"
        )

    # --- 4. derived version gate ------------------------------------------
    if derived_version == INPUT_VERSION:
        raise DeriveFailure("derived version must differ from the input version")
    if derived_version != DERIVED_VERSION:
        raise DeriveFailure(
            f"derived version must be exactly {DERIVED_VERSION!r} this round, "
            f"got {derived_version!r}"
        )

    new_dist = f"{INPUT_DISTRIBUTION}-{derived_version}.dist-info"
    out_name = f"{INPUT_DISTRIBUTION}-{derived_version}-{in_tags}.whl"
    patch_files = parse_patch_files(patch)
    patch_sha256 = sha256_file(patch)

    # --- 5. unpack OUTSIDE the repository ---------------------------------
    root_guard = repo_root()
    workdir = Path(tempfile.mkdtemp(prefix="hcu-derive-"))
    try:
        if root_guard is not None:
            import os

            guard = str(root_guard.resolve())
            if str(workdir.resolve()).startswith(guard + os.sep):
                raise DeriveFailure("mktemp directory landed inside the repository")
        unpack_root = workdir / "unpacked"
        with zipfile.ZipFile(input_wheel) as zin:
            zin.extractall(unpack_root)

        # --- 6. patch check then apply ------------------------------------
        check = run_git_apply(unpack_root, patch, "--check")
        if check.returncode != 0:
            raise DeriveFailure(f"patch --check failed: {check.stderr.strip()}")
        applied = run_git_apply(unpack_root, patch)
        if applied.returncode != 0:
            raise DeriveFailure(f"patch failed: {applied.stderr.strip()}")

        # --- 7. metadata surgery -------------------------------------------
        metadata_path = unpack_root / in_dist / "METADATA"
        metadata_path.write_bytes(
            rewrite_metadata_version(
                metadata_path.read_bytes(), INPUT_VERSION, derived_version
            )
        )
        (unpack_root / in_dist).rename(unpack_root / new_dist)
        package_version_updated = maybe_rewrite_package_version(
            unpack_root, INPUT_VERSION, derived_version
        )

        # --- 8. RECORD -----------------------------------------------------
        (unpack_root / new_dist / "RECORD").write_bytes(
            rebuild_record(unpack_root, new_dist)
        )

        # --- 9. deterministic wheel ----------------------------------------
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / out_name
        build_wheel(out_path, input_wheel, unpack_root, in_dist, new_dist)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    # --- 10. strict gates + provenance --------------------------------------
    try:
        report = verify(input_wheel, out_path, patch, expected_input_sha256)
    except GateFailure as failure:
        out_path.unlink(missing_ok=True)
        raise DeriveFailure(f"derived wheel rejected by the strict gates: {failure}")

    provenance = {
        "input_wheel": input_wheel.name,
        "input_sha256": expected_input_sha256,
        "patch": patch.name,
        "patch_sha256": patch_sha256,
        "patch_files": patch_files,
        "derived_wheel": out_name,
        "derived_version": derived_version,
        "derived_sha256": report["derived_sha256"],
        "package_version_source_updated": package_version_updated,
        "source_commit": source_commit,
        "deterministic_packaging": (
            "input member order; per-member date/mode/compression preserved; "
            "DEFLATE level 9; sorted RECORD"
        ),
    }
    provenance_path = out_path.with_suffix(".provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive the Hygon HCU SGLang wheel from the pinned FlagOS wheel"
    )
    parser.add_argument("--input-wheel", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--derived-version", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    try:
        provenance = derive(
            args.input_wheel,
            args.expected_input_sha256,
            args.patch,
            args.output_dir,
            args.derived_version,
            args.source_commit,
        )
    except DeriveFailure as failure:
        print(f"DERIVE-FAILURE: {failure}", file=sys.stderr)
        return 1
    print("DERIVED-WHEEL::" + json.dumps(provenance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
