# Copyright (c) 2026 BAAI. All rights reserved.

"""Static and executable contracts for the Ascend 0.5.18 image recipe."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest


_ROOT = Path(__file__).parents[2]
_DOCKERFILE = _ROOT / "docker" / "ascend" / "empty-0.5.18.containerfile"
_DOCKERIGNORE = Path(f"{_DOCKERFILE}.dockerignore")
_ASCEND_CONFIG = _ROOT / ".github" / "configs" / "ascend.yml"
_VERIFIER_PATH = _ROOT / ".github" / "scripts" / "ascend" / "verify_environment.py"
_SPEC = importlib.util.spec_from_file_location(
    "ascend_verify_environment", _VERIFIER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_VERIFIER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _VERIFIER
_SPEC.loader.exec_module(_VERIFIER)


def test_vendor_runtime_versions_match_the_real_empty_image() -> None:
    assert _VERIFIER.EXPECTED_RUNTIME_DISTRIBUTIONS["torch"] == "2.8.0+cpu"
    assert _VERIFIER.EXPECTED_RUNTIME_DISTRIBUTIONS["triton-ascend"] == "3.2.1"

    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")
    assert "'torch==2.8.0+cpu'" in dockerfile
    assert "'triton-ascend==3.2.1'" in dockerfile
    assert "'torch':'2.8.0+cpu'" in dockerfile
    assert "'triton-ascend':'3.2.1'" in dockerfile


def test_source_archive_versions_are_guarded() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")

    assert "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=${SGLANG_VERSION}" in dockerfile
    assert "SETUPTOOLS_SCM_PRETEND_VERSION=${SGLANG_VERSION}" not in dockerfile
    assert "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLAG_GEMS" not in dockerfile
    assert "old='version = \\\"5.3.0rc2\\\"'" in dockerfile
    assert "expected one FlagGems version anchor" in dockerfile
    assert "actual=m.version('sglang')" in dockerfile
    assert "actual != '${SGLANG_VERSION}'" in dockerfile
    assert "actual=m.version('flag-gems')" in dockerfile
    assert "actual != '${FLAGGEMS_VERSION}'" in dockerfile
    assert _VERIFIER.EXPECTED_RUNTIME_DISTRIBUTIONS["sglang"] == "0.5.18"
    assert _VERIFIER.EXPECTED_RUNTIME_DISTRIBUTIONS["flag-gems"] == "5.3.0"


def test_image_contract_checks_survive_python_optimized_mode() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")

    assert "python3 -c" in dockerfile
    assert "assert " not in dockerfile
    assert "sys.exit(" in dockerfile


def test_kernel_patcher_is_part_of_the_image_context_and_build() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")
    dockerignore = _DOCKERIGNORE.read_text(encoding="utf-8")

    assert "COPY scripts/ascend/patch_sgl_kernel_npu.py" in dockerfile
    assert "python3 /usr/local/bin/patch-sgl-kernel-npu" in dockerfile
    assert "not hasattr(tl, 'insert_slice')" in dockerfile
    assert "sys.exit('triton-ascend insert_slice API mismatch')" in dockerfile
    assert "!scripts/ascend/patch_sgl_kernel_npu.py" in dockerignore


def test_source_trees_are_recreated_instead_of_overlaid() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")

    assert "rm -rf /opt/sglang-0.5.18" in dockerfile
    assert "rm -rf /opt/FlagGems" in dockerfile
    assert "rm -rf /opt/sglang-plugin-fl" in dockerfile


def test_kernel_patch_provenance_is_labeled() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")

    assert _VERIFIER.EXPECTED_SOLVE_TRIL_SHA256 in dockerfile
    assert "ai.flagos.sgl-kernel-npu.solve-tril.sha256" in dockerfile


def test_ci_uses_the_ascend_runtime_without_privileged_mode() -> None:
    config = _ASCEND_CONFIG.read_text(encoding="utf-8")
    options = config.split("container_options:", maxsplit=1)[1]

    assert "--runtime=ascend" in options
    assert "ASCEND_VISIBLE_DEVICES=0,1,2,3" in options
    assert "ASCEND_RT_VISIBLE_DEVICES=0,1,2,3" in options
    assert "--privileged" not in options
    assert "--device /dev/" not in options


class _Distribution:
    def __init__(self, root: Path):
        self.root = root
        self.files = [Path("sgl_kernel_npu-2026.5.1.dist-info/RECORD")]

    def locate_file(self, relative_path: Path) -> Path:
        return self.root / relative_path


def _solve_tril_source(*, old: int, new: int) -> str:
    lines = ["import triton.language.extra.cann.extension as al"]
    lines.extend(f"new_{index} = al.insert_slice(" for index in range(new))
    lines.extend(f"old_{index} = tl.insert_slice(" for index in range(old))
    return "\n".join(lines) + "\n"


def _write_kernel_installation(tmp_path: Path, *, old: int, new: int) -> Path:
    path = tmp_path / _VERIFIER.SOLVE_TRIL_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    content = _solve_tril_source(old=old, new=new).encode()
    path.write_bytes(content)
    record = tmp_path / "sgl_kernel_npu-2026.5.1.dist-info" / "RECORD"
    record.parent.mkdir()
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
    record.write_text(
        f"{_VERIFIER.SOLVE_TRIL_RELATIVE_PATH.as_posix()},"
        f"sha256={digest.decode('ascii')},{len(content)}\n",
        encoding="utf-8",
    )
    return path


def test_verifier_accepts_the_fully_patched_kernel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_kernel_installation(tmp_path, old=0, new=15)
    monkeypatch.setattr(
        _VERIFIER,
        "EXPECTED_SOLVE_TRIL_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        _VERIFIER.metadata,
        "distribution",
        lambda name: _Distribution(tmp_path),
    )

    _VERIFIER._require_solve_tril_patch()


def test_verifier_rejects_unexpected_patched_source_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_kernel_installation(tmp_path, old=0, new=15)
    monkeypatch.setattr(
        _VERIFIER.metadata,
        "distribution",
        lambda name: _Distribution(tmp_path),
    )

    with pytest.raises(RuntimeError, match="digest mismatch"):
        _VERIFIER._require_solve_tril_patch()


def test_verifier_rejects_stale_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_kernel_installation(tmp_path, old=0, new=15)
    monkeypatch.setattr(
        _VERIFIER,
        "EXPECTED_SOLVE_TRIL_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    record = tmp_path / "sgl_kernel_npu-2026.5.1.dist-info" / "RECORD"
    record.write_text(
        f"{_VERIFIER.SOLVE_TRIL_RELATIVE_PATH.as_posix()},sha256=stale,1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        _VERIFIER.metadata,
        "distribution",
        lambda name: _Distribution(tmp_path),
    )

    with pytest.raises(RuntimeError, match="RECORD does not describe"):
        _VERIFIER._require_solve_tril_patch()


@pytest.mark.parametrize(("old", "new"), [(1, 15), (0, 14), (0, 16)])
def test_verifier_rejects_any_other_kernel_call_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    old: int,
    new: int,
) -> None:
    _write_kernel_installation(tmp_path, old=old, new=new)
    monkeypatch.setattr(
        _VERIFIER.metadata,
        "distribution",
        lambda name: _Distribution(tmp_path),
    )

    with pytest.raises(RuntimeError, match="patch mismatch"):
        _VERIFIER._require_solve_tril_patch()
