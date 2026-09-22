#!/usr/bin/env python3
"""Patch the pinned sgl-kernel-npu solve_tril source for Triton Ascend 3.2.1."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import stat
import tempfile
from importlib import metadata
from pathlib import Path


SOLVE_TRIL_RELATIVE_PATH = Path("sgl_kernel_npu/fla/solve_tril.py")
REQUIRED_AL_IMPORT = b"import triton.language.extra.cann.extension as al"
STALE_CALL = b"tl.insert_slice("
ASCEND_CALL = b"al.insert_slice("
EXPECTED_STALE_CALLS = 6
EXPECTED_ORIGINAL_ASCEND_CALLS = 9
EXPECTED_PATCHED_ASCEND_CALLS = 15


class PatchValidationError(RuntimeError):
    """Raised when the installed wheel does not match the pinned source contract."""


def locate_installation() -> tuple[Path, Path]:
    """Return solve_tril.py and RECORD from the installed distribution."""

    try:
        distribution = metadata.distribution("sgl-kernel-npu")
    except metadata.PackageNotFoundError as exc:
        raise PatchValidationError("sgl-kernel-npu is not installed") from exc
    record_candidates = [
        path
        for path in distribution.files or ()
        if str(path).replace("\\", "/").endswith(".dist-info/RECORD")
    ]
    if len(record_candidates) != 1:
        raise PatchValidationError(
            "sgl-kernel-npu must contain exactly one dist-info/RECORD, "
            f"found {len(record_candidates)}"
        )
    return (
        Path(distribution.locate_file(SOLVE_TRIL_RELATIVE_PATH)),
        Path(distribution.locate_file(record_candidates[0])),
    )


def _atomic_write(path: Path, content: bytes) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary_path: Path | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def patch_solve_tril(path: str | Path) -> Path:
    """Atomically replace the six stale Triton calls in the pinned wheel source."""

    target = Path(path).resolve(strict=True)
    original = target.read_bytes()

    if REQUIRED_AL_IMPORT not in original:
        raise PatchValidationError(
            f"{target} does not import the Triton Ascend extension as al"
        )

    stale_count = original.count(STALE_CALL)
    ascend_count = original.count(ASCEND_CALL)
    if stale_count == 0 and ascend_count == EXPECTED_PATCHED_ASCEND_CALLS:
        return target
    if (
        stale_count != EXPECTED_STALE_CALLS
        or ascend_count != EXPECTED_ORIGINAL_ASCEND_CALLS
    ):
        raise PatchValidationError(
            f"{target} must contain old/new insert_slice counts "
            f"{EXPECTED_STALE_CALLS}/{EXPECTED_ORIGINAL_ASCEND_CALLS} before "
            f"patching or 0/{EXPECTED_PATCHED_ASCEND_CALLS} after patching; "
            f"found {stale_count}/{ascend_count}"
        )

    patched = original.replace(STALE_CALL, ASCEND_CALL)
    if patched.count(STALE_CALL) != 0:
        raise PatchValidationError(f"failed to replace every stale call in {target}")
    if patched.count(ASCEND_CALL) != EXPECTED_PATCHED_ASCEND_CALLS:
        raise PatchValidationError(f"unexpected al.insert_slice count in {target}")

    _atomic_write(target, patched)

    return target


def invalidate_bytecode(target_path: str | Path) -> list[Path]:
    """Remove stale solve_tril bytecode after the same-size source rewrite."""

    target = Path(target_path).resolve(strict=True)
    candidates = list((target.parent / "__pycache__").glob(f"{target.stem}.*.pyc"))
    candidates.append(target.with_suffix(".pyc"))
    removed: list[Path] = []
    for candidate in candidates:
        if candidate.is_file():
            candidate.unlink()
            removed.append(candidate)
    return removed


def update_record(record_path: str | Path, target_path: str | Path) -> Path:
    """Update the wheel RECORD hash/size after the guarded source patch."""

    record = Path(record_path).resolve(strict=True)
    target = Path(target_path).resolve(strict=True)
    rows = list(csv.reader(io.StringIO(record.read_text(encoding="utf-8"))))
    relative_name = SOLVE_TRIL_RELATIVE_PATH.as_posix()
    matching = [row for row in rows if row and row[0] == relative_name]
    if len(matching) != 1:
        raise PatchValidationError(
            f"{record} must contain exactly one {relative_name} row, "
            f"found {len(matching)}"
        )

    content = target.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
    matching[0][1] = f"sha256={digest.decode('ascii')}"
    matching[0][2] = str(len(content))
    bytecode_prefix = f"{SOLVE_TRIL_RELATIVE_PATH.parent.as_posix()}/__pycache__/"
    legacy_bytecode = SOLVE_TRIL_RELATIVE_PATH.with_suffix(".pyc").as_posix()
    rows = [
        row
        for row in rows
        if not row
        or not (
            (
                row[0].startswith(bytecode_prefix)
                and Path(row[0]).name.startswith(f"{target.stem}.")
                and row[0].endswith(".pyc")
            )
            or row[0] == legacy_bytecode
        )
    ]
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    _atomic_write(record, output.getvalue().encode("utf-8"))
    return record


def main() -> int:
    target, record = locate_installation()
    target = patch_solve_tril(target)
    invalidate_bytecode(target)
    update_record(record, target)
    print(f"[ascend-kernel-patch] verified compatible solve_tril source at {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
