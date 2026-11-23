#!/usr/bin/env python3
"""Strict diff gates for the derived Hygon SGLang wheel.

Compares the derived wheel against the SHA-pinned original wheel plus the
reviewed HCU patch and enforces that NOTHING outside the reviewed allowlist
differs:

  allowed to change
    - the SGLang Python files the patch actually modifies;
    - the METADATA ``Version`` line (input -> derived local version);
    - the ``*.dist-info`` directory name;
    - RECORD (rebuilt by the derive script; every hash re-validated here);
    - the in-package version source, only when the original wheel's
      ``sglang/_version.py`` carried the input distribution version.

  must stay byte-identical
    - every native payload file (.so/.a/.h/...);
    - every Python file the patch does not modify;
    - Requires-Dist / Provides-Extra / every other METADATA header;
    - entry_points.txt, WHEEL, top_level.txt, license files;
    - the wheel tags (python/abi/platform).

Additional structural proofs
    - the forward patch no longer applies to the derived wheel, while the
      reverse patch passes ``git apply --check``;
    - patching a fresh unpack of the ORIGINAL wheel reproduces the derived
      wheel's patched files byte-for-byte;
    - the derived RECORD's hash/size for every member matches the archive;
    - filename / METADATA / dist-info directory / package version are
      mutually consistent.

Exit code 0 means every gate passed. ``--json PATH`` additionally writes the
machine-readable report (used as disposable evidence; never committed).

Usage:
  verify_derived_sglang_wheel.py --input-wheel W --expected-input-sha256 H \
      --derived-wheel W [--expected-derived-sha256 H] --patch P \
      [--json PATH]
"""

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

INPUT_DISTRIBUTION = "sglang"
INPUT_VERSION = "0.5.18+flagos"
NATIVE_SUFFIXES = (".so", ".a", ".h", ".hpp", ".bc", ".hsaco")


class GateFailure(Exception):
    """A strict diff gate was violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_patch_files(patch: Path) -> list:
    """Wheel-relative paths the patch modifies (its ``+++ b/...`` side).

    The HCU patch is written against a ``python/`` checkout prefix and
    applied with ``git apply -p2`` (a/python/sglang/... -> sglang/...).
    """
    files = []
    for line in patch.read_text(encoding="utf-8").splitlines():
        if line.startswith("+++ b/"):
            rel = line[len("+++ b/") :].strip()
            parts = rel.split("/", 2)
            if len(parts) < 3 or parts[0] != "python" or parts[1] != "sglang":
                raise GateFailure(
                    f"patch touches a non-sglang/python path: {rel} "
                    "(this verifier only handles python/sglang/* patches)"
                )
            files.append("sglang/" + parts[2])
    if not files:
        raise GateFailure("patch modifies no file")
    return sorted(set(files))


def wheel_dist_info(archive: zipfile.ZipFile) -> str:
    """The one top-level ``*.dist-info`` directory of the wheel.

    Wheels need not carry explicit directory entries, so the directory is
    reconstructed from the files directly inside it.
    """
    dirs = {
        name.split("/", 1)[0]
        for name in archive.namelist()
        if name.count("/") == 1 and name.split("/", 1)[0].endswith(".dist-info")
    }
    if len(dirs) != 1:
        raise GateFailure(
            f"expected exactly one top-level .dist-info/, got {sorted(dirs)}"
        )
    return dirs.pop()


def read_dist_info_file(archive: zipfile.ZipFile, dist_info: str, name: str) -> bytes:
    return archive.read(f"{dist_info}/{name}")


def parse_wheel_filename(filename: str) -> tuple:
    stem = filename.removesuffix(".whl")
    parts = stem.split("-")
    if len(parts) < 3 or len(parts) > 5:
        raise GateFailure(f"cannot parse wheel filename {filename!r}")
    return parts[0], parts[1], "-".join(parts[2:])


def run_git_apply(
    workdir: Path, patch: Path, *args: str
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "apply", *args, "-p2", str(patch)],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        check=False,
    )


def unpack(archive_path: Path, target: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(target)


def collect_files(root: Path) -> dict:
    return {
        str(path.relative_to(root)): path
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def verify(
    input_wheel: Path,
    derived_wheel: Path,
    patch: Path,
    expected_input_sha256: str,
    expected_derived_sha256=None,
) -> dict:
    """Run every gate; raise GateFailure on the first violation."""

    # git apply runs with the unpack directories as cwd: paths must be
    # absolute before any tempdir exists
    input_wheel, derived_wheel, patch = (
        input_wheel.resolve(),
        derived_wheel.resolve(),
        patch.resolve(),
    )

    # --- G0: artifact identity -------------------------------------------
    if sha256_file(input_wheel) != expected_input_sha256:
        raise GateFailure("input wheel SHA256 does not match the pinned value")
    derived_sha256 = sha256_file(derived_wheel)
    if expected_derived_sha256 and derived_sha256 != expected_derived_sha256:
        raise GateFailure(
            f"derived wheel SHA256 {derived_sha256} != expected "
            f"{expected_derived_sha256}"
        )
    patch_sha256 = sha256_file(patch)

    with zipfile.ZipFile(input_wheel) as zin, zipfile.ZipFile(derived_wheel) as zout:
        in_dist, out_dist = wheel_dist_info(zin), wheel_dist_info(zout)

        # --- G1/G7: metadata headers -------------------------------------
        in_meta = BytesParser().parsebytes(
            read_dist_info_file(zin, in_dist, "METADATA")
        )
        out_meta = BytesParser().parsebytes(
            read_dist_info_file(zout, out_dist, "METADATA")
        )
        if (in_meta.get("Name") or "").lower() != INPUT_DISTRIBUTION:
            raise GateFailure(f"input wheel is not {INPUT_DISTRIBUTION!r}")
        if in_meta.get("Version") != INPUT_VERSION:
            raise GateFailure(
                f"input wheel version {in_meta.get('Version')!r} != {INPUT_VERSION!r}"
            )
        derived_version = out_meta.get("Version")
        if derived_version == INPUT_VERSION:
            raise GateFailure("derived wheel did not change the version")

        header_diffs = sorted(
            {
                key
                for key in set(in_meta) | set(out_meta)
                if key != "Version" and in_meta.get_all(key) != out_meta.get_all(key)
            }
        )
        if header_diffs:
            raise GateFailure(f"METADATA headers drifted: {header_diffs}")
        requires_identical = in_meta.get_all("Requires-Dist") == out_meta.get_all(
            "Requires-Dist"
        )

        # --- G2: filename / dist-info / METADATA version consistency ----
        in_name, in_version, in_tags = parse_wheel_filename(input_wheel.name)
        out_name, out_version, out_tags = parse_wheel_filename(derived_wheel.name)
        if in_version != INPUT_VERSION or (in_name or "").lower() != INPUT_DISTRIBUTION:
            raise GateFailure(f"input wheel filename mis-parsed: {input_wheel.name}")
        if (out_name or "").lower() != INPUT_DISTRIBUTION:
            raise GateFailure(f"derived wheel name {out_name!r} != sglang")
        if out_version != derived_version:
            raise GateFailure(
                f"derived filename version {out_version!r} != METADATA "
                f"{derived_version!r}"
            )
        if out_tags != in_tags:
            raise GateFailure(f"wheel tags drifted: {in_tags!r} -> {out_tags!r}")
        expected_dist = f"{INPUT_DISTRIBUTION}-{derived_version}.dist-info"
        if out_dist != expected_dist:
            raise GateFailure(
                f"dist-info directory {out_dist!r} != expected {expected_dist!r}"
            )

        # --- G3-G6: payload comparison on fresh unpacks ------------------
        patch_files = parse_patch_files(patch)
        with tempfile.TemporaryDirectory(prefix="hcu-verify-") as tmp:
            tmp = Path(tmp)
            orig_root, derived_root, patched_ref = (
                tmp / "orig",
                tmp / "derived",
                tmp / "ref",
            )
            unpack(input_wheel, orig_root)
            unpack(derived_wheel, derived_root)

            # reference tree: original + forward patch
            patched_ref.mkdir()
            for rel, path in collect_files(orig_root).items():
                target = patched_ref / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
            check = run_git_apply(patched_ref, patch, "--check")
            if check.returncode != 0:
                raise GateFailure(
                    f"patch does not apply to the original: {check.stderr}"
                )
            applied = run_git_apply(patched_ref, patch)
            if applied.returncode != 0:
                raise GateFailure(f"patch failed on the original: {applied.stderr}")

            # version-source policy: the package version file only changes
            # when the ORIGINAL carried the input distribution version in it
            in_version_py = orig_root / "sglang" / "_version.py"
            out_version_py = derived_root / "sglang" / "_version.py"
            version_source_updated = False
            if in_version_py.is_file() or out_version_py.is_file():
                if not (in_version_py.is_file() and out_version_py.is_file()):
                    raise GateFailure(
                        "sglang/_version.py presence differs between wheels"
                    )
                in_text = in_version_py.read_text(encoding="utf-8")
                out_text = out_version_py.read_text(encoding="utf-8")
                if f"__version__ = version = '{INPUT_VERSION}'" in in_text:
                    version_source_updated = True
                    if f"__version__ = version = '{derived_version}'" not in out_text:
                        raise GateFailure(
                            "sglang/_version.py carried the input version but "
                            "was not rewritten to the derived version"
                        )
                elif in_version_py.read_bytes() != out_version_py.read_bytes():
                    raise GateFailure(
                        "sglang/_version.py changed although the original "
                        "wheel's package version did not carry the "
                        "distribution version"
                    )

            orig_files = collect_files(orig_root)
            derived_files = collect_files(derived_root)

            def normalize(rel: str, dist: str) -> str:
                # both trees ship exactly one dist-info directory whose NAME
                # is an allowed difference; normalize it away for the member
                # set comparison
                if rel.startswith(dist + "/"):
                    return "<dist-info>/" + rel.split("/", 1)[1]
                return rel

            orig_norm = {normalize(rel, in_dist): rel for rel in orig_files}
            derived_norm = {normalize(rel, out_dist): rel for rel in derived_files}
            if set(orig_norm) != set(derived_norm):
                only_orig = sorted(set(orig_norm) - set(derived_norm))
                only_derived = sorted(set(derived_norm) - set(orig_norm))
                raise GateFailure(
                    f"member sets differ; only in original: {only_orig[:5]}; "
                    f"only in derived: {only_derived[:5]}"
                )

            unchanged_differ, patched_reproduced, native_report = [], [], {}
            for norm, in_rel in sorted(orig_norm.items()):
                out_rel = derived_norm[norm]
                in_bytes = orig_files[in_rel].read_bytes()
                out_bytes = derived_files[out_rel].read_bytes()
                is_patched = in_rel in patch_files
                if is_patched:
                    ref_bytes = (patched_ref / in_rel).read_bytes()
                    if out_bytes != ref_bytes:
                        patched_reproduced.append(in_rel)
                    continue
                if in_rel.endswith(".dist-info/METADATA"):
                    # only the Version line may differ (headers checked above;
                    # this catches raw-byte drift the parser would normalize)
                    in_lines = in_bytes.split(b"\n")
                    out_lines = out_bytes.split(b"\n")
                    version_lines = [
                        i for i, ln in enumerate(in_lines) if ln.startswith(b"Version:")
                    ]
                    if len(version_lines) != 1:
                        raise GateFailure(
                            "original METADATA has an unexpected Version-line count"
                        )
                    i = version_lines[0]
                    rest_equal = in_lines[:i] + in_lines[i + 1 :] == out_lines[
                        :i
                    ] + out_lines[i + 1 :] and len(in_lines) == len(out_lines)
                    if not rest_equal or not out_lines[i].startswith(b"Version:"):
                        unchanged_differ.append(in_rel)
                    continue
                if in_rel.endswith(".dist-info/RECORD"):
                    continue  # rebuilt by design; validated below
                if in_bytes != out_bytes:
                    unchanged_differ.append(in_rel)
                if in_rel.endswith(NATIVE_SUFFIXES):
                    native_report[in_rel] = (
                        hashlib.sha256(in_bytes).hexdigest()
                        == hashlib.sha256(out_bytes).hexdigest()
                    )
            if unchanged_differ:
                raise GateFailure(
                    f"files outside the allowlist drifted: {unchanged_differ[:10]}"
                )
            if patched_reproduced:
                raise GateFailure(
                    f"patched files do not equal original+patch: {patched_reproduced}"
                )
            if not all(native_report.values()):
                bad = [k for k, ok in native_report.items() if not ok]
                raise GateFailure(f"native payload changed: {bad}")

            # --- G9: patch direction proofs on the derived tree ----------
            forward = run_git_apply(derived_root, patch, "--check")
            if forward.returncode == 0:
                raise GateFailure(
                    "the forward patch still applies to the derived wheel"
                )
            reverse = run_git_apply(derived_root, patch, "--reverse", "--check")
            if reverse.returncode != 0:
                raise GateFailure(
                    f"reverse patch does not check against the derived wheel: "
                    f"{reverse.stderr}"
                )

            # --- G10: derived RECORD hash/size validity -------------------
            record_rel = f"{out_dist}/RECORD"
            record_entries = {}
            for line in (
                derived_files[record_rel].read_text(encoding="utf-8").splitlines()
            ):
                if not line:
                    continue
                # paths may contain literal commas (e.g. triton config
                # JSONs like lora_expand,K=1024,...); the digest and size
                # fields never do, so split from the right
                fields = line.rsplit(",", 2)
                if len(fields) != 3:
                    raise GateFailure(f"malformed RECORD line: {line!r}")
                record_entries[fields[0]] = (fields[1], fields[2])
            if set(record_entries) != set(derived_files):
                missing = sorted(set(derived_files) - set(record_entries))
                extra = sorted(set(record_entries) - set(derived_files))
                raise GateFailure(
                    f"RECORD member set mismatch; missing: {missing[:5]}; "
                    f"extra: {extra[:5]}"
                )
            for rel, path in derived_files.items():
                if rel == record_rel:
                    if record_entries[rel] != ("", ""):
                        raise GateFailure("RECORD's own entry must carry no hash")
                    continue
                digest, size = record_entries[rel]
                data = path.read_bytes()
                expected_digest = (
                    base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                    .rstrip(b"=")
                    .decode("ascii")
                )
                # RECORD stores the digest as "sha256=<urlsafe-b64-nopad>"
                bare_digest = digest.removeprefix("sha256=")
                if bare_digest != expected_digest or size != str(len(data)):
                    raise GateFailure(f"RECORD hash/size wrong for {rel}")

    return {
        "input_sha256": expected_input_sha256,
        "derived_sha256": derived_sha256,
        "patch_sha256": patch_sha256,
        "patch_files": patch_files,
        "member_sets_equal": True,
        "unchanged_files_differ": [],
        "native_payload": {
            "files": sorted(native_report),
            "byte_identical": True,
        },
        "patched_files_reproduced": patch_files,
        "metadata_headers_differ": [],
        "entry_points_identical": True,
        "wheel_metadata_identical": True,
        "top_level_identical": True,
        "licenses_identical": True,
        "version_consistency": {
            "filename": out_version,
            "metadata": derived_version,
            "dist_info": out_dist,
            "package_version_source_updated": version_source_updated,
        },
        "forward_patch_rejected": True,
        "reverse_patch_applies": True,
        "record_valid": True,
        "requires_dist_identical": requires_identical,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strict diff gates for the derived Hygon SGLang wheel"
    )
    parser.add_argument("--input-wheel", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--derived-wheel", type=Path, required=True)
    parser.add_argument("--expected-derived-sha256", default=None)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument(
        "--json", type=Path, default=None, help="write the machine-readable report here"
    )
    args = parser.parse_args()

    try:
        report = verify(
            args.input_wheel,
            args.derived_wheel,
            args.patch,
            args.expected_input_sha256,
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
