# Copyright (c) 2026 BAAI. All rights reserved.

"""Platform configuration loader for sglang-plugin-FL tests."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PLATFORMS_DIR = Path(__file__).resolve().parents[1] / "platforms"


def _load_structured(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        data = json.loads(text)
    return data if isinstance(data, dict) else {}


@dataclass
class Tolerance:
    rtol: float = 1e-5
    atol: float = 1e-8
    exact: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Tolerance":
        return cls(
            rtol=float(raw.get("rtol", 1e-5)),
            atol=float(raw.get("atol", 1e-8)),
            exact=bool(raw.get("exact", False)),
        )


@dataclass
class TestFilter:
    include: str | list[str] = "*"
    exclude: list[str] = field(default_factory=list)


@dataclass
class E2eTestConfig:
    """E2E test cases grouped by CI stage, task, and model.

    Ported from vllm-plugin-FL #581: the ``e2e`` block in each device's YAML
    is keyed by CI stage (``pr`` / ``nightly`` / ``weekly``).
    """

    # stage -> task -> model -> list of case names
    # e.g. {"pr": {"inference": {"qwen3": ["4b_tp2"]}},
    #        "nightly": {"inference": {...}, "concurrent": {...}}}
    tests: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    def get_cases(
        self,
        stage: str | None = None,
        task: str | None = None,
        model: str | None = None,
        test_list: str | None = None,
    ) -> list[dict[str, str]]:
        """Return a flat list of {task, model, case} dicts, optionally filtered.

        Args:
            stage: CI stage key (``"pr"``, ``"nightly"``, ``"weekly"``).
                   When ``None``, cases from all stages are returned.
            task: Filter by task name.
            model: Filter by model name.
            test_list: Comma-separated list of case names to include.
        """
        allowed = {t.strip() for t in test_list.split(",")} if test_list else None
        cases: list[dict[str, str]] = []
        for stage_key, task_models in self.tests.items():
            if stage and stage_key != stage:
                continue
            if not isinstance(task_models, dict):
                continue
            for task_name, models in task_models.items():
                if task and task_name != task:
                    continue
                if not isinstance(models, dict):
                    continue
                for model_name, case_list in models.items():
                    if model and model_name != model:
                        continue
                    if not isinstance(case_list, list):
                        case_list = [case_list]
                    for case in case_list:
                        if allowed and str(case) not in allowed:
                            continue
                        cases.append(
                            {"task": task_name, "model": model_name, "case": str(case)}
                        )
        return cases


@dataclass
class PlatformConfig:
    platform: str
    vendor: str
    device_types: dict[str, Any]
    tolerance: dict[str, dict[str, Tolerance]]
    engine_overrides: dict[str, Any]
    device_overrides: dict[str, Any]
    env_defaults: dict[str, str]
    unsupported_features: list[str]
    device_tests: dict[str, dict[str, Any]]
    device: str
    # Platform declares its runtime cannot host concurrent engines: e2e
    # cases run serially even with many devices (see musa.yaml — MCCL dies
    # when two TP groups initialize at once; ported from vllm-plugin-FL #581).
    e2e_serial: bool = False

    @classmethod
    def load(
        cls,
        platform: str,
        device: str | None = None,
        platforms_dir: Path | None = None,
    ) -> "PlatformConfig":
        platforms_dir = platforms_dir or _PLATFORMS_DIR
        path = _resolve_platform_file(platform, platforms_dir)
        raw = _load_structured(path)
        device_types = raw.get("device_types", {}) or {}
        if isinstance(device_types, list):
            device_types = {str(name): {} for name in device_types}

        active_device = device or next(iter(device_types), "")
        device_tests = {
            str(name): raw.get(str(name), {})
            for name in device_types
            if isinstance(raw.get(str(name), {}), dict)
        }
        tolerance: dict[str, dict[str, Tolerance]] = {}
        for category, dtypes in (raw.get("tolerance") or {}).items():
            tolerance[category] = {}
            for dtype_name, tol_raw in dtypes.items():
                tolerance[category][dtype_name] = Tolerance.from_dict(tol_raw)

        return cls(
            platform=raw.get("platform", path.stem),
            vendor=raw.get("vendor", ""),
            device_types=device_types,
            tolerance=tolerance,
            engine_overrides=_ensure_mapping(
                raw.get("engine_overrides"), "engine_overrides"
            ),
            device_overrides=raw.get("device_overrides", {}) or {},
            env_defaults=raw.get("env_defaults", {}) or {},
            unsupported_features=raw.get("unsupported_features", []) or [],
            device_tests=device_tests,
            device=active_device,
            e2e_serial=bool(raw.get("e2e_serial", False)),
        )

    def apply_env_defaults(self) -> None:
        for key, value in self.env_defaults.items():
            os.environ.setdefault(str(key), str(value))

    def get_unit_filter(self) -> TestFilter:
        raw = self._tests_section("unit")
        return TestFilter(
            include=raw.get("include", "*"), exclude=raw.get("exclude", [])
        )

    def get_e2e_tests(self) -> E2eTestConfig:
        """Return e2e test configuration (stage-grouped) for the active device."""
        raw = self._tests_section("e2e")
        return E2eTestConfig(tests=raw)

    def get_functional_filter(self) -> TestFilter:
        raw = self._tests_section("functional")
        return TestFilter(
            include=raw.get("include", "*"), exclude=raw.get("exclude", [])
        )

    def get_benchmark_tests(self, stage: str = "nightly") -> dict[str, Any]:
        """Return benchmark test configuration for the active device and CI stage."""
        benchmark = self._tests_section("benchmark")
        # New-style: keyed by stage (pr / nightly / weekly)
        if "pr" in benchmark or "nightly" in benchmark:
            return benchmark.get(stage) or benchmark.get("nightly", {})
        # Legacy flat format — return as-is for backwards compatibility
        return benchmark

    def get_engine_overrides(self, model: str, case: str) -> dict[str, Any]:
        """Return SGLang engine overrides for one model case.

        Device-specific values take precedence over platform-wide values.
        The returned flat mapping is merged into the common model config later.
        """
        model_overrides = _ensure_mapping(
            self.engine_overrides.get(model),
            f"engine_overrides.{model}",
        )
        overrides = _ensure_mapping(
            model_overrides.get(case),
            f"engine_overrides.{model}.{case}",
        )

        device_override = self.device_overrides.get(self.device, {})
        if isinstance(device_override, dict):
            device_engine_overrides = _ensure_mapping(
                device_override.get("engine_overrides"),
                f"device_overrides.{self.device}.engine_overrides",
            )
            device_model_overrides = _ensure_mapping(
                device_engine_overrides.get(model),
                f"device_overrides.{self.device}.engine_overrides.{model}",
            )
            overrides.update(
                _ensure_mapping(
                    device_model_overrides.get(case),
                    f"device_overrides.{self.device}.engine_overrides.{model}.{case}",
                )
            )

        return overrides

    def get_tolerance(
        self, category: str = "inference", dtype: str = "default"
    ) -> Tolerance:
        device_override = self.device_overrides.get(self.device, {})
        if isinstance(device_override, dict):
            device_tolerance = device_override.get("tolerance", {}).get(category, {})
            if dtype in device_tolerance:
                return Tolerance.from_dict(device_tolerance[dtype])
            if "default" in device_tolerance:
                return Tolerance.from_dict(device_tolerance["default"])

        category_tolerance = self.tolerance.get(category, {})
        if dtype in category_tolerance:
            return category_tolerance[dtype]
        if "default" in category_tolerance:
            return category_tolerance["default"]

        return Tolerance()

    def should_skip_model(self, model_name: str) -> bool:
        return any(token in model_name for token in self.unsupported_features)

    def _tests_section(self, name: str) -> dict[str, Any]:
        return (
            self.device_tests.get(self.device, {}).get("tests", {}).get(name, {}) or {}
        )


def _ensure_mapping(raw: Any, field_name: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be a mapping, got {type(raw).__name__}")
    return dict(raw)


def _resolve_platform_file(platform: str, platforms_dir: Path) -> Path:
    direct = platforms_dir / f"{platform}.yaml"
    if direct.exists():
        return direct

    for candidate in platforms_dir.glob("*.yaml"):
        if candidate.stem == "template":
            continue
        raw = _load_structured(candidate)
        device_types = raw.get("device_types", {}) or {}
        names = device_types if isinstance(device_types, list) else device_types.keys()
        if platform in names:
            return candidate

    available = ", ".join(sorted(p.stem for p in platforms_dir.glob("*.yaml")))
    raise FileNotFoundError(
        f"Platform config not found for {platform}. Available: {available}"
    )
