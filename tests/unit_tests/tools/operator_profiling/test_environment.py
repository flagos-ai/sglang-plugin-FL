# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import pytest
import torch

from tools.operator_profiling import environment


@pytest.fixture
def valid_environment(monkeypatch, tmp_path):
    repository_root = tmp_path / "checkout"
    (repository_root / "sglang_fl").mkdir(parents=True)
    model_path = tmp_path / "model"
    model_path.mkdir()

    versions = {
        "sglang": "0.5.11",
        "sglang_fl": "0.1.0",
        "torch": "2.11.0+cu130",
        "transformers": "5.6.0",
        "flagtree": "0.6.2a1",
    }
    monkeypatch.setattr(
        environment, "_distribution_version", lambda name: versions.get(name)
    )
    monkeypatch.setattr(
        environment,
        "_entry_point_values",
        lambda group, name: [environment._REQUIRED_ENTRY_POINTS[group]],
    )

    origins = {
        "sglang": Path("/opt/sglang/sglang/__init__.py"),
        "sglang_fl": repository_root / "sglang_fl" / "__init__.py",
        "torch": Path("/opt/python/torch/__init__.py"),
        "triton": Path("/opt/python/triton/__init__.py"),
    }
    monkeypatch.setattr(environment, "_module_origin", origins.get)
    monkeypatch.setattr(
        environment.metadata,
        "packages_distributions",
        lambda: {"triton": ["flagtree"]},
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"GPU-{index}")
    monkeypatch.setattr(
        environment,
        "_inspect_engine_api",
        lambda: {
            "importable": True,
            "class": "sglang.srt.entrypoints.engine.Engine",
            "profiling_methods": ["start_profile", "stop_profile"],
            "sglang_runtime_version": versions["sglang"],
        },
    )
    return repository_root, model_path, versions


def test_valid_environment_returns_auditable_report(valid_environment):
    repository_root, model_path, _ = valid_environment

    report = environment.validate_profiling_environment(
        model_path=model_path,
        tp_size=4,
        repository_root=repository_root,
        require_validated_versions=True,
    )

    assert report["version_mismatches"] == {}
    assert report["triton_namespace_providers"] == ["flagtree"]
    assert report["cuda"] == {
        "available": True,
        "device_count": 4,
        "device_names": ["GPU-0", "GPU-1", "GPU-2", "GPU-3"],
    }
    assert report["warnings"] == []
    assert report["engine_api"]["importable"] is True


def test_version_mismatch_is_warning_by_default(monkeypatch, valid_environment):
    repository_root, model_path, versions = valid_environment
    versions["sglang"] = "0.5.12"
    monkeypatch.setattr(
        environment, "_distribution_version", lambda name: versions.get(name)
    )

    report = environment.validate_profiling_environment(
        model_path=model_path,
        tp_size=4,
        repository_root=repository_root,
    )

    assert report["version_mismatches"] == {
        "sglang": {"expected": "0.5.11", "actual": "0.5.12"}
    }
    assert "unvalidated version combination" in report["warnings"][0]


def test_strict_version_check_rejects_mismatch(monkeypatch, valid_environment):
    repository_root, model_path, versions = valid_environment
    versions["flagtree"] = "0.6.3"
    monkeypatch.setattr(
        environment, "_distribution_version", lambda name: versions.get(name)
    )

    with pytest.raises(environment.ProfilingEnvironmentError, match="version mismatch"):
        environment.validate_profiling_environment(
            model_path=model_path,
            tp_size=4,
            repository_root=repository_root,
            require_validated_versions=True,
        )


def test_rejects_non_flagtree_triton_namespace(monkeypatch, valid_environment):
    repository_root, model_path, _ = valid_environment
    monkeypatch.setattr(
        environment.metadata,
        "packages_distributions",
        lambda: {"triton": ["triton"]},
    )

    with pytest.raises(
        environment.ProfilingEnvironmentError,
        match="not provided by FlagTree",
    ):
        environment.validate_profiling_environment(
            model_path=model_path,
            tp_size=4,
            repository_root=repository_root,
        )


def test_reports_all_hard_prerequisite_failures(monkeypatch, valid_environment):
    repository_root, model_path, _ = valid_environment
    model_path.rmdir()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(environment, "_entry_point_values", lambda group, name: [])

    with pytest.raises(environment.ProfilingEnvironmentError) as raised:
        environment.validate_profiling_environment(
            model_path=model_path,
            tp_size=4,
            repository_root=repository_root,
        )

    message = str(raised.value)
    assert "model directory not found" in message
    assert "missing entry point sglang.srt.platforms" in message
    assert "missing entry point sglang.srt.plugins" in message
    assert "CUDA is not available" in message
    assert "requires at least 4 visible GPUs" in message


def test_rejects_sglang_engine_import_failure(monkeypatch, valid_environment):
    repository_root, model_path, _ = valid_environment

    def fail_import():
        raise ImportError("incompatible xgrammar")

    monkeypatch.setattr(environment, "_inspect_engine_api", fail_import)

    with pytest.raises(
        environment.ProfilingEnvironmentError,
        match="source and installed dependencies may not match",
    ):
        environment.validate_profiling_environment(
            model_path=model_path,
            tp_size=4,
            repository_root=repository_root,
        )
