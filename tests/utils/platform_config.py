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
class FunctionalTests:
    tests: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    def get_cases(self, task: str | None = None, model: str | None = None) -> list[dict[str, str]]:
        cases: list[dict[str, str]] = []
        for task_name, models in self.tests.items():
            if task and task_name != task:
                continue
            for model_name, case_list in models.items():
                if model and model_name != model:
                    continue
                for case in case_list:
                    cases.append({"task": task_name, "model": model_name, "case": case})
        return cases


@dataclass
class PlatformConfig:
    platform: str
    vendor: str
    device_types: dict[str, Any]
    tolerance: dict[str, dict[str, Tolerance]]
    device_overrides: dict[str, Any]
    env_defaults: dict[str, str]
    unsupported_features: list[str]
    device_tests: dict[str, dict[str, Any]]
    device: str

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
            device_overrides=raw.get("device_overrides", {}) or {},
            env_defaults=raw.get("env_defaults", {}) or {},
            unsupported_features=raw.get("unsupported_features", []) or [],
            device_tests=device_tests,
            device=active_device,
        )

    def apply_env_defaults(self) -> None:
        for key, value in self.env_defaults.items():
            os.environ.setdefault(str(key), str(value))

    def get_unit_filter(self) -> TestFilter:
        raw = self._tests_section("unit")
        return TestFilter(include=raw.get("include", "*"), exclude=raw.get("exclude", []))

    def get_e2e_tests(self) -> FunctionalTests:
        raw = self._tests_section("e2e")
        return FunctionalTests(tests=raw)

    def get_functional_filter(self) -> TestFilter:
        raw = self._tests_section("functional")
        return TestFilter(include=raw.get("include", "*"), exclude=raw.get("exclude", []))

    def get_benchmark_tests(self) -> dict[str, Any]:
        return self._tests_section("benchmark")

    def get_tolerance(self, category: str = "inference", dtype: str = "default") -> Tolerance:
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
        return self.device_tests.get(self.device, {}).get("tests", {}).get(name, {}) or {}


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
    raise FileNotFoundError(f"Platform config not found for {platform}. Available: {available}")


