# Copyright (c) 2026 BAAI. All rights reserved.

"""Tests for the guarded sgl-kernel-npu source compatibility patch."""

from __future__ import annotations

import base64
import importlib.util
import csv
import hashlib
import io
import os
import sys
import py_compile
from pathlib import Path

import pytest


_PATCHER_PATH = (
    Path(__file__).parents[2] / "scripts" / "ascend" / "patch_sgl_kernel_npu.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "ascend_patch_sgl_kernel_npu", _PATCHER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_PATCHER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PATCHER
_SPEC.loader.exec_module(_PATCHER)


def _source(*, stale: int = 6, ascend: int = 9, include_import: bool = True) -> bytes:
    lines = ["import triton.language as tl"]
    if include_import:
        lines.append("import triton.language.extra.cann.extension as al")
    lines.extend(f"existing_{index} = al.insert_slice(" for index in range(ascend))
    lines.extend(f"stale_{index} = tl.insert_slice(" for index in range(stale))
    return ("\n".join(lines) + "\n").encode()


def _importable_source() -> bytes:
    lines = [
        "if False:",
        "    import triton.language.extra.cann.extension as al",
        "CALLS = [",
    ]
    lines.extend('    "al.insert_slice(",' for _ in range(9))
    lines.extend('    "tl.insert_slice(",' for _ in range(6))
    lines.append("]")
    return ("\n".join(lines) + "\n").encode()


def test_exact_pinned_wheel_shape_is_patched_atomically(tmp_path: Path) -> None:
    path = tmp_path / "solve_tril.py"
    path.write_bytes(_source())

    result = _PATCHER.patch_solve_tril(path)

    assert result == path.resolve()
    patched = path.read_bytes()
    assert patched.count(_PATCHER.STALE_CALL) == 0
    assert patched.count(_PATCHER.ASCEND_CALL) == 15
    assert not list(tmp_path.glob(".solve_tril.py.*.tmp"))


def test_fully_patched_source_is_an_idempotent_noop(tmp_path: Path) -> None:
    path = tmp_path / "solve_tril.py"
    original = _source(stale=0, ascend=15)
    path.write_bytes(original)

    result = _PATCHER.patch_solve_tril(path)

    assert result == path.resolve()
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".solve_tril.py.*.tmp"))


def test_record_is_updated_for_the_patched_file(tmp_path: Path) -> None:
    path = tmp_path / "solve_tril.py"
    path.write_bytes(_source())
    record = tmp_path / "RECORD"
    record.write_text(
        "sgl_kernel_npu/fla/solve_tril.py,sha256=stale,1\n"
        "sgl_kernel_npu/__init__.py,sha256=unchanged,2\n",
        encoding="utf-8",
    )

    _PATCHER.patch_solve_tril(path)
    _PATCHER.update_record(record, path)

    rows = list(csv.reader(io.StringIO(record.read_text(encoding="utf-8"))))
    assert rows[0][0] == "sgl_kernel_npu/fla/solve_tril.py"
    expected_digest = base64.urlsafe_b64encode(
        hashlib.sha256(path.read_bytes()).digest()
    ).rstrip(b"=")
    assert rows[0][1] == f"sha256={expected_digest.decode('ascii')}"
    assert rows[0][2] == str(path.stat().st_size)
    assert rows[1] == ["sgl_kernel_npu/__init__.py", "sha256=unchanged", "2"]
    assert not list(tmp_path.glob(".RECORD.*.tmp"))


def test_stale_same_size_bytecode_is_removed_before_import(tmp_path: Path) -> None:
    path = tmp_path / "solve_tril.py"
    path.write_bytes(_importable_source())
    bytecode = Path(
        py_compile.compile(
            str(path),
            doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
        )
    )
    record = tmp_path / "RECORD"
    record.write_text(
        "sgl_kernel_npu/fla/solve_tril.py,sha256=stale,1\n"
        f"sgl_kernel_npu/fla/__pycache__/{bytecode.name},sha256=stale,2\n",
        encoding="utf-8",
    )

    _PATCHER.patch_solve_tril(path)
    removed = _PATCHER.invalidate_bytecode(path)
    _PATCHER.update_record(record, path)

    assert bytecode in removed
    assert not bytecode.exists()
    assert "__pycache__/solve_tril." not in record.read_text(encoding="utf-8")
    spec = importlib.util.spec_from_file_location("patched_solve_tril", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.CALLS == ["al.insert_slice("] * 15


def test_idempotent_rerun_removes_stale_bytecode_and_preserves_others(
    tmp_path: Path,
) -> None:
    path = tmp_path / "solve_tril.py"
    original = _importable_source()
    path.write_bytes(original)
    fixed_mtime = int(path.stat().st_mtime)
    os.utime(path, (fixed_mtime, fixed_mtime))
    bytecode = Path(
        py_compile.compile(
            str(path),
            doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
        )
    )
    unrelated = bytecode.parent / "other.cpython-311.pyc"
    unrelated.write_bytes(b"keep")
    path.write_bytes(original.replace(_PATCHER.STALE_CALL, _PATCHER.ASCEND_CALL))
    os.utime(path, (fixed_mtime, fixed_mtime))

    _PATCHER.patch_solve_tril(path)
    removed = _PATCHER.invalidate_bytecode(path)

    assert removed == [bytecode]
    assert unrelated.read_bytes() == b"keep"
    spec = importlib.util.spec_from_file_location("repatched_solve_tril", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.CALLS == ["al.insert_slice("] * 15


def test_missing_record_row_is_rejected_without_modification(tmp_path: Path) -> None:
    path = tmp_path / "solve_tril.py"
    path.write_bytes(_source(stale=0, ascend=15))
    record = tmp_path / "RECORD"
    original = b"sgl_kernel_npu/__init__.py,sha256=unchanged,2\n"
    record.write_bytes(original)

    with pytest.raises(_PATCHER.PatchValidationError, match="exactly one"):
        _PATCHER.update_record(record, path)

    assert record.read_bytes() == original
    assert not list(tmp_path.glob(".RECORD.*.tmp"))


@pytest.mark.parametrize(
    ("stale_count", "ascend_count"),
    [(5, 9), (7, 9), (6, 8), (6, 10), (0, 14), (0, 16)],
)
def test_unexpected_call_shape_is_rejected_without_modification(
    tmp_path: Path, stale_count: int, ascend_count: int
) -> None:
    path = tmp_path / "solve_tril.py"
    original = _source(stale=stale_count, ascend=ascend_count)
    path.write_bytes(original)

    with pytest.raises(_PATCHER.PatchValidationError, match="old/new insert_slice"):
        _PATCHER.patch_solve_tril(path)

    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".solve_tril.py.*.tmp"))


def test_missing_al_import_is_rejected_without_modification(tmp_path: Path) -> None:
    path = tmp_path / "solve_tril.py"
    original = _source(include_import=False)
    path.write_bytes(original)

    with pytest.raises(_PATCHER.PatchValidationError, match="does not import"):
        _PATCHER.patch_solve_tril(path)

    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".solve_tril.py.*.tmp"))
