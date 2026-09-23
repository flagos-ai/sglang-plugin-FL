# Copyright (c) 2026 BAAI. All rights reserved.

"""Static and executable contracts for the Ascend 0.5.18 image recipe."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


_ROOT = Path(__file__).parents[2]
_DOCKERFILE = _ROOT / "docker" / "ascend" / "empty-0.5.18.containerfile"
_DOCKERIGNORE = Path(f"{_DOCKERFILE}.dockerignore")
_ASCEND_CONFIG = _ROOT / ".github" / "configs" / "ascend.yml"
_ASCEND_CHECK = _ROOT / ".github" / "scripts" / "ascend" / "check.sh"
_PLATFORM_REGISTRY = _ROOT / ".github" / "configs" / "platforms.yml"
_ASCEND_TEST_CONFIG = _ROOT / "tests" / "platforms" / "ascend.yaml"
_PLATFORM_UNIT_TESTS = _ROOT / "tests" / "unit_tests" / "platform"
_FUNCTIONAL_WORKFLOW = _ROOT / ".github" / "workflows" / "_functional_test.yml"
_COMPAT_PROBE_PATH = _ROOT / "examples" / "ascend_compat_probe.py"
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


def test_official_kernel_release_is_pinned_without_source_patching() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")
    dockerignore = _DOCKERIGNORE.read_text(encoding="utf-8")

    assert "ARG SGL_KERNEL_NPU_RELEASE=2026.8.10" in dockerfile
    assert (
        "ARG SGL_KERNEL_NPU_COMMIT=e05fc90e85041d6080431c8ce543ab112c56e16d"
        in dockerfile
    )
    assert "ARG SGL_KERNEL_NPU_VERSION=2026.6.1" in dockerfile
    assert (
        "ARG SGL_KERNEL_NPU_SHA256="
        "fa4ad5afa6bb748d683da5ee8a9cc702c13d0398d3db53651b4bb79e3f8d0e95" in dockerfile
    )
    assert (
        "sgl-kernel-npu-2026.8.10-torch2.8.0-py311-cann8.5.0-a3-aarch64.zip"
        in dockerfile
    )
    assert "deep_ep-${DEEP_EP_VERSION}-cp311-cp311-linux_aarch64.whl" in dockerfile
    assert "sgl_kernel_npu-${SGL_KERNEL_NPU_VERSION}" in dockerfile
    assert "torch_memory_saver-0.0.8" in dockerfile
    assert "ln -sf deep_ep/deep_ep_cpp*.so" in dockerfile
    assert "patch_sgl_kernel_npu" not in dockerfile
    assert "patch_sgl_kernel_npu" not in dockerignore
    assert "not hasattr(tl, 'insert_slice')" in dockerfile
    assert "sys.exit('triton-ascend insert_slice API mismatch')" in dockerfile


def test_source_trees_are_recreated_instead_of_overlaid() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")

    assert "rm -rf /opt/sglang-0.5.18" in dockerfile
    assert "rm -rf /opt/FlagGems" in dockerfile
    assert "rm -rf /opt/sglang-plugin-fl" in dockerfile


def test_plugin_source_revision_is_required_and_labeled() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")

    assert "ARG SGLANG_FL_REVISION" in dockerfile
    assert 'test -n "${SGLANG_FL_REVISION}"' in dockerfile
    assert "/opt/sglang-plugin-fl/.flagos-source-commit" in dockerfile
    assert "ai.flagos.sglang-plugin-fl.revision" in dockerfile
    assert '--build-arg SGLANG_FL_REVISION="$(git rev-parse HEAD)"' in dockerfile


def test_kernel_release_provenance_is_labeled() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")

    assert _VERIFIER.EXPECTED_SOLVE_TRIL_SHA256 in dockerfile
    assert "ai.flagos.sgl-kernel-npu.release" in dockerfile
    assert "ai.flagos.sgl-kernel-npu.revision" in dockerfile
    assert "ai.flagos.sgl-kernel-npu.version" in dockerfile
    assert "ai.flagos.sgl-kernel-npu.archive.sha256" in dockerfile
    assert "ai.flagos.sgl-kernel-npu.solve-tril.sha256" in dockerfile
    assert "ai.flagos.deep-ep.version" in dockerfile


def test_ci_uses_the_ascend_runtime_without_privileged_mode() -> None:
    config = _ASCEND_CONFIG.read_text(encoding="utf-8")
    options = config.split("container_options:", maxsplit=1)[1]

    assert "--runtime=ascend" in options
    assert "ASCEND_VISIBLE_DEVICES=0,1,2,3" in options
    assert "ASCEND_RT_VISIBLE_DEVICES=0,1,2,3" in options
    assert "GLOO_SOCKET_IFNAME=lo" in options
    assert "HCCL_HOST_SOCKET_PORT_RANGE=auto" in options
    assert "HCCL_NPU_SOCKET_PORT_RANGE=auto" in options
    assert "HCCL_IF_BASE_PORT=" not in options
    assert "--privileged" not in options
    assert "--device /dev/" not in options


def test_ci_mounts_host_models_and_driver_read_only() -> None:
    config = yaml.safe_load(_ASCEND_CONFIG.read_text(encoding="utf-8"))
    volumes = config["container_volumes"]

    assert "/mnt/airs-business/cicd/models:/data/models/Qwen:ro" in volumes
    assert "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro" in volumes
    assert "/usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro" in volumes
    assert "/etc/ascend_install.info:/etc/ascend_install.info:ro" in volumes
    assert "/var/queue_schedule:/var/queue_schedule" in volumes
    assert "/var/queue_schedule:/var/queue_schedule:ro" not in volumes


def test_ascend_check_ignores_empty_npu_smi_stub() -> None:
    script = _ASCEND_CHECK.read_text(encoding="utf-8")

    assert '[ -x "$path_hit" ] && [ -s "$path_hit" ]' in script
    assert 'if [ -x "$cand" ] && [ -s "$cand" ]; then' in script
    assert script.index("    /usr/local/bin/npu-smi \\") < script.index(
        "    /usr/local/sbin/npu-smi \\"
    )


def test_ascend_unit_scope_excludes_musa_autopatch_tests() -> None:
    config = yaml.safe_load(_ASCEND_TEST_CONFIG.read_text(encoding="utf-8"))
    excluded = set(config["910c"]["tests"]["unit"]["exclude"])
    musa_tests = {
        path.relative_to(_ROOT / "tests" / "unit_tests").as_posix()
        for path in _PLATFORM_UNIT_TESTS.glob("test_musa_*.py")
    }

    assert musa_tests
    assert musa_tests <= excluded


def test_enabled_ascend_ci_requires_an_immutable_image_digest() -> None:
    registry = yaml.safe_load(_PLATFORM_REGISTRY.read_text(encoding="utf-8"))
    config = yaml.safe_load(_ASCEND_CONFIG.read_text(encoding="utf-8"))
    enabled = registry["platforms"]["ascend"]["enabled"]
    image = config["ci_image"]

    digest_reference = re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", image)
    assert not enabled or digest_reference is not None, (
        "Ascend CI may be enabled only after ci_image is pinned to its "
        "registry-reported sha256 digest"
    )
    if enabled:
        assert image.startswith("harbor.baai.ac.cn/plugin/sglang-plugin-fl@sha256:")


def test_functional_matrix_runs_the_real_ascend_compatibility_probe() -> None:
    workflow = _FUNCTIONAL_WORKFLOW.read_text(encoding="utf-8")
    probe = _COMPAT_PROBE_PATH.read_text(encoding="utf-8")

    assert "- name: Run Ascend compatibility probe" in workflow
    assert "if: inputs.platform == 'ascend'" in workflow
    assert "run: python3 examples/ascend_compat_probe.py" in workflow
    assert "MTP_SOURCE_SHAPE = (1, 1, 2, 2, 128, 128)" in probe
    assert "LOGSUMEXP_SHAPE = (2, 16384)" in probe
    assert "kernel_module.move_intermediate_cache(" in probe
    assert "row_logsumexp_topk(" in probe


class _Distribution:
    def __init__(self, root: Path):
        self.root = root
        self.files = [Path("sgl_kernel_npu-2026.6.1.dist-info/RECORD")]

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
    record = tmp_path / "sgl_kernel_npu-2026.6.1.dist-info" / "RECORD"
    record.parent.mkdir()
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
    record.write_text(
        f"{_VERIFIER.SOLVE_TRIL_RELATIVE_PATH.as_posix()},"
        f"sha256={digest.decode('ascii')},{len(content)}\n",
        encoding="utf-8",
    )
    return path


def test_verifier_accepts_the_official_kernel_source(
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

    _VERIFIER._require_solve_tril_source()


def test_verifier_rejects_unexpected_kernel_source_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_kernel_installation(tmp_path, old=0, new=15)
    monkeypatch.setattr(
        _VERIFIER.metadata,
        "distribution",
        lambda name: _Distribution(tmp_path),
    )

    with pytest.raises(RuntimeError, match="digest mismatch"):
        _VERIFIER._require_solve_tril_source()


def test_verifier_rejects_stale_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_kernel_installation(tmp_path, old=0, new=15)
    monkeypatch.setattr(
        _VERIFIER,
        "EXPECTED_SOLVE_TRIL_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    record = tmp_path / "sgl_kernel_npu-2026.6.1.dist-info" / "RECORD"
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
        _VERIFIER._require_solve_tril_source()


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

    with pytest.raises(RuntimeError, match="source mismatch"):
        _VERIFIER._require_solve_tril_source()


class _Schema:
    def __init__(self, value: str):
        self.value = value

    def __str__(self) -> str:
        return self.value


def _torch_with_causal_conv1d_schema(schema: str) -> SimpleNamespace:
    default = SimpleNamespace(_schema=_Schema(schema))
    causal_conv1d = SimpleNamespace(default=default)
    return SimpleNamespace(
        ops=SimpleNamespace(npu=SimpleNamespace(causal_conv1d=causal_conv1d))
    )


def test_verifier_accepts_the_sglang_0_5_18_causal_conv1d_abi() -> None:
    torch = _torch_with_causal_conv1d_schema(_VERIFIER.EXPECTED_CAUSAL_CONV1D_SCHEMA)

    _VERIFIER._require_causal_conv1d_abi(torch)


def test_verifier_rejects_the_legacy_causal_conv1d_abi() -> None:
    legacy = (
        "npu::causal_conv1d(Tensor x, Tensor weight, Tensor conv_states, "
        "Tensor query_start_loc, Tensor cache_indices, Tensor has_initial_state, "
        "Tensor? bias=None, bool activation_mode=False, int pad_slot_id=-1) -> Tensor"
    )
    torch = _torch_with_causal_conv1d_schema(legacy)

    with pytest.raises(RuntimeError, match="ABI mismatch"):
        _VERIFIER._require_causal_conv1d_abi(torch)


def test_verifier_rejects_a_missing_causal_conv1d_op() -> None:
    torch = SimpleNamespace(ops=SimpleNamespace(npu=SimpleNamespace()))

    with pytest.raises(RuntimeError, match="is not registered"):
        _VERIFIER._require_causal_conv1d_abi(torch)


def test_verifier_blocks_on_the_real_packed_vision_padding_probe() -> None:
    source = _VERIFIER_PATH.read_text(encoding="utf-8")

    assert "_sglang_fl_unaligned_head_fallback" in source
    assert "VisionAscendAttention" in source
    assert "sequence_ends = (3, 8)" in source
    assert "max_error > 0.02 or mean_error > 0.002" in source
    assert "_require_vision_padding_numerics(torch)" in source
