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

"""Fail-fast environment checks for runtime operator profiling."""

from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
from pathlib import Path
from typing import Any


VALIDATED_VERSIONS = {
    "sglang": "0.5.11",
    "torch": "2.11.0+cu130",
    "flagtree": "0.6.2a1",
}

_REQUIRED_ENTRY_POINTS = {
    "sglang.srt.platforms": "sglang_fl:activate_platform",
    "sglang.srt.plugins": "sglang_fl:load_plugin",
}


class ProfilingEnvironmentError(RuntimeError):
    """Raised when the profiling command cannot run in the active environment."""


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _entry_point_values(group: str, name: str) -> list[str]:
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        matches = entry_points.select(group=group, name=name)
    else:  # pragma: no cover - Python 3.8 compatibility
        matches = [
            entry_point
            for entry_point in entry_points.get(group, [])
            if entry_point.name == name
        ]
    return sorted(entry_point.value for entry_point in matches)


def _module_origin(name: str) -> Path | None:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve()


def _is_relative_to(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _inspect_engine_api() -> dict[str, Any]:
    """Import the actual Engine entry point and verify profiler controls."""

    import sglang
    from sglang.srt.entrypoints.engine import Engine

    missing = [
        name
        for name in ("start_profile", "stop_profile")
        if not callable(getattr(Engine, name, None))
    ]
    if missing:
        raise RuntimeError(f"SGLang Engine is missing methods: {missing}")
    return {
        "importable": True,
        "class": f"{Engine.__module__}.{Engine.__qualname__}",
        "profiling_methods": ["start_profile", "stop_profile"],
        "sglang_runtime_version": str(sglang.__version__),
    }


def validate_profiling_environment(
    *,
    model_path: str | Path,
    tp_size: int,
    repository_root: str | Path | None = None,
    require_validated_versions: bool = False,
) -> dict[str, Any]:
    """Validate hard runtime prerequisites and return auditable metadata.

    Version differences are reported but only rejected when
    ``require_validated_versions`` is set. Structural requirements such as the
    plugin entry points, FlagTree-owned ``triton`` namespace, model directory,
    and visible GPU count are always enforced.
    """

    errors: list[str] = []
    warnings: list[str] = []
    resolved_model_path = Path(model_path).expanduser().resolve()
    if not resolved_model_path.is_dir():
        errors.append(f"model directory not found: {resolved_model_path}")
    if tp_size <= 0:
        errors.append(f"tp_size must be positive, got {tp_size}")

    versions = {name: _distribution_version(name) for name in VALIDATED_VERSIONS}
    versions["sglang_fl"] = _distribution_version("sglang_fl")
    versions["transformers"] = _distribution_version("transformers")
    for name in ("sglang", "sglang_fl", "torch", "transformers", "flagtree"):
        if versions.get(name) is None:
            errors.append(f"required distribution is not installed: {name}")

    entry_points: dict[str, list[str]] = {}
    for group, expected_value in _REQUIRED_ENTRY_POINTS.items():
        values = _entry_point_values(group, "sglang_fl")
        entry_points[group] = values
        if expected_value not in values:
            errors.append(
                f"missing entry point {group}: sglang_fl={expected_value}; "
                "install this checkout with `python -m pip install -e . --no-deps`"
            )

    module_origins = {
        name: str(origin) if (origin := _module_origin(name)) is not None else None
        for name in ("sglang", "sglang_fl", "torch", "triton")
    }
    for name, origin in module_origins.items():
        if origin is None:
            errors.append(f"required Python module is not importable: {name}")

    if repository_root is not None and module_origins["sglang_fl"] is not None:
        expected_package = Path(repository_root).resolve() / "sglang_fl"
        actual_origin = Path(module_origins["sglang_fl"])
        if not _is_relative_to(actual_origin, expected_package):
            errors.append(
                "sglang_fl resolves outside this checkout: "
                f"{actual_origin}; expected a path under {expected_package}"
            )

    triton_providers = sorted(metadata.packages_distributions().get("triton", []))
    if "flagtree" not in {
        provider.lower().replace("-", "_") for provider in triton_providers
    }:
        errors.append(
            "the `triton` import namespace is not provided by FlagTree; "
            f"detected providers: {triton_providers or ['none']}"
        )

    cuda: dict[str, Any] = {"available": False, "device_count": 0, "device_names": []}
    try:
        import torch

        # PyTorch wheels can keep the CUDA local-version suffix only in the
        # imported module version (for example ``2.11.0+cu130``), while the
        # distribution metadata reports ``2.11.0``.  The runtime module is the
        # authoritative value for a compatibility check.
        versions["torch"] = str(torch.__version__)
        cuda["available"] = bool(torch.cuda.is_available())
        cuda["device_count"] = int(torch.cuda.device_count())
        if cuda["available"]:
            cuda["device_names"] = [
                torch.cuda.get_device_name(index)
                for index in range(cuda["device_count"])
            ]
        if not hasattr(torch, "profiler"):
            errors.append("active PyTorch build does not provide torch.profiler")
        if not cuda["available"]:
            errors.append("CUDA is not available to the active Python process")
        if tp_size > 0 and cuda["device_count"] < tp_size:
            errors.append(
                f"tp_size={tp_size} requires at least {tp_size} visible GPUs, "
                f"found {cuda['device_count']}"
            )
    except Exception as error:
        errors.append(f"failed to inspect PyTorch/CUDA: {error}")

    engine_api: dict[str, Any] = {"importable": False}
    if cuda["available"] and module_origins["sglang"] is not None:
        try:
            engine_api = _inspect_engine_api()
            versions["sglang"] = engine_api["sglang_runtime_version"]
        except Exception as error:
            errors.append(
                "cannot import the SGLang Engine used by profiling; the SGLang "
                f"source and installed dependencies may not match: {error}"
            )

    mismatches = {
        name: {"expected": expected, "actual": versions.get(name)}
        for name, expected in VALIDATED_VERSIONS.items()
        if versions.get(name) != expected
    }
    if mismatches:
        mismatch_text = ", ".join(
            f"{name}={item['actual']!r} (validated {item['expected']!r})"
            for name, item in mismatches.items()
        )
        if require_validated_versions:
            errors.append(f"version mismatch: {mismatch_text}")
        else:
            warnings.append(f"unvalidated version combination: {mismatch_text}")

    report = {
        "model_path": str(resolved_model_path),
        "tp_size": tp_size,
        "versions": versions,
        "validated_versions": dict(VALIDATED_VERSIONS),
        "version_mismatches": mismatches,
        "module_origins": module_origins,
        "entry_points": entry_points,
        "triton_namespace_providers": triton_providers,
        "cuda": cuda,
        "engine_api": engine_api,
        "warnings": warnings,
    }
    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise ProfilingEnvironmentError(
            "operator profiling environment check failed:\n" + details
        )
    return report


__all__ = [
    "ProfilingEnvironmentError",
    "VALIDATED_VERSIONS",
    "validate_profiling_environment",
]
