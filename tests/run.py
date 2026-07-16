#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.

"""Unified test entry point for sglang-plugin-FL.

- tests/platforms/<platform>.yaml selects scopes/cases per device.
- tests/models/<model>/<case>.yaml stores model and engine settings.
- tests/benchmarks/configs/smoke.yaml stores benchmark smoke templates.

Examples:
    python tests/run.py --platform cuda --device a100 --scope unit
    python tests/run.py --platform cuda --device a100 --scope functional
    python tests/run.py --platform cuda --device a100 --scope benchmark
    python tests/run.py --platform cuda --device a100 --scope functional
    python tests/run.py --platform cuda --device a100 --scope benchmark --benchmark latency
    python tests/run.py --platform cuda --device a100 --scope functional
    python tests/run.py --platform cuda --device a100 --scope benchmark --model qwen3 --case 06b_tp1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DIR = _REPO_ROOT / "tests"
sys.path.insert(0, str(_REPO_ROOT))

from tests.utils.model_config import ModelConfig
from tests.utils.platform_config import PlatformConfig


def load_structured_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        data = json.loads(text)
    return data if isinstance(data, dict) else {}


@dataclass
class TestCase:
    name: str
    pytest_path: str
    task: str
    model: str = ""
    case: str = ""
    extra_args: list[str] = field(default_factory=list)
    extra_env: dict[str, str] = field(default_factory=dict)


class TestRunner:
    def __init__(
        self,
        platform: str,
        device: str | None,
        scope: str,
        task: str | None = None,
        model: str | None = None,
        case: str | None = None,
        benchmark: str | None = None,
        extra_pytest_args: list[str] | None = None,
    ) -> None:
        self.config = PlatformConfig.load(platform, device)
        self.scope = scope
        self.task = task
        self.model = model
        self.case = case
        self.benchmark = benchmark
        self.extra_pytest_args = extra_pytest_args or []

    def run(self) -> int:
        self.config.apply_env_defaults()
        os.environ["FL_TEST_PLATFORM"] = self.config.platform
        os.environ["FL_TEST_DEVICE"] = self.config.device

        cases = self.discover_tests()
        if not cases:
            print("[run] No test cases selected.")
            return 0

        print(f"[run] Platform: {self.config.platform}")
        print(f"[run] Device:   {self.config.device}")
        print(f"[run] Scope:    {self.scope}")
        print(f"[run] Cases:    {len(cases)}")
        print()

        failed = 0
        for case in cases:
            rc = self._run_single(case)
            if rc != 0:
                failed += 1

        print()
        print(f"[run] Passed: {len(cases) - failed}, Failed: {failed}")
        return 1 if failed else 0

    def discover_tests(self) -> list[TestCase]:
        cases: list[TestCase] = []
        if self.scope in ("all", "unit"):
            cases.extend(self._discover_unit_tests())
        if self.scope in ("all", "e2e"):
            cases.extend(self._discover_e2e_tests())
        if self.scope in ("all", "functional"):
            cases.extend(self._discover_functional_tests())
        if self.scope in ("all", "benchmark"):
            cases.extend(self._discover_benchmark_tests())
        return cases

    def _discover_unit_tests(self) -> list[TestCase]:
        unit_filter = self.config.get_unit_filter()
        extra_args = ["-q", "--tb=short"]
        for pattern in unit_filter.exclude:
            extra_args.extend(["--ignore", f"tests/unit_tests/{pattern}"])
        if unit_filter.include != "*" and isinstance(unit_filter.include, list):
            extra_args.extend(["-k", " or ".join(unit_filter.include)])
        return [TestCase(name="unit", pytest_path="tests/unit_tests", task="unit", extra_args=extra_args)]

    def _discover_e2e_tests(self) -> list[TestCase]:
        cases: list[TestCase] = []

        # Future-compatible structured e2e discovery.
        for item in self.config.get_e2e_tests().get_cases(task=self.task, model=self.model):
            model_name = item["model"]
            case_name = item["case"]
            if self.case and self.case != case_name:
                continue
            if self.config.should_skip_model(model_name):
                print(f"[run] Skipping e2e/{model_name}/{case_name} (unsupported feature)")
                continue
            task = item["task"]
            test_path = f"tests/e2e_tests/{task}/test_{task}_smoke.py"
            if not (_REPO_ROOT / test_path).exists():
                print(f"[run] Warning: structured e2e test not found: {test_path}")
                continue
            cases.append(
                TestCase(
                    name=f"e2e/{task}/{model_name}/{case_name}",
                    pytest_path=test_path,
                    task="e2e",
                    model=model_name,
                    case=case_name,
                    extra_args=["-v", "--tb=short", "-s"],
                    extra_env={"FL_TEST_MODEL": model_name, "FL_TEST_CASE": case_name},
                )
            )
        return cases

    def _discover_functional_tests(self) -> list[TestCase]:
        functional_filter = self.config.get_functional_filter()
        extra_args = ["-v", "--tb=short", "-s"]
        for pattern in functional_filter.exclude:
            extra_args.extend(["--ignore", f"tests/functional_tests/{pattern}"])
        if functional_filter.include != "*" and isinstance(functional_filter.include, list):
            extra_args.extend(["-k", " or ".join(functional_filter.include)])

        root = _REPO_ROOT / "tests" / "functional_tests"
        if not root.exists():
            return []

        return [
            TestCase(
                name="functional",
                pytest_path="tests/functional_tests",
                task="functional",
                extra_args=extra_args,
            )
        ]
    def _discover_benchmark_tests(self) -> list[TestCase]:
        benchmark_cfg = self.config.get_benchmark_tests()
        if not benchmark_cfg.get("enabled", False):
            return []

        selected = benchmark_cfg.get("smoke", [])
        if isinstance(selected, str):
            selected_types = {selected}
        else:
            selected_types = set(selected)
        if self.benchmark:
            selected_types &= {self.benchmark}

        config_path = _TESTS_DIR / "benchmarks" / "configs" / "smoke.yaml"
        smoke_config = load_structured_config(config_path)

        cases: list[TestCase] = []
        for bench_type, case_list in smoke_config.items():
            if bench_type not in selected_types:
                continue
            if not isinstance(case_list, list):
                continue
            pytest_path = f"tests/benchmarks/test_benchmark_{bench_type}.py"
            if not (_REPO_ROOT / pytest_path).exists():
                print(f"[run] Benchmark test file missing: {pytest_path}")
                continue

            for raw_case in case_list:
                runtime_case = dict(raw_case)
                model_name = runtime_case.pop("model", self.model)
                case_name = runtime_case.pop("case", self.case)
                if not model_name or not case_name:
                    print(
                        f"[run] Skipping benchmark/{bench_type}: "
                        "missing model/case in smoke.yaml and no --model/--case provided"
                    )
                    continue
                model_name = str(model_name)
                case_name = str(case_name)
                if self.model and self.model != model_name:
                    continue
                if self.case and self.case != case_name:
                    continue
                if self.config.should_skip_model(model_name):
                    print(f"[run] Skipping benchmark/{model_name}/{case_name} (unsupported feature)")
                    continue

                model_cfg = ModelConfig.load(model_name, case_name)
                self._inject_model_config(runtime_case, model_cfg)
                name = str(runtime_case.get("name", f"{bench_type}_smoke"))
                cases.append(
                    TestCase(
                        name=f"benchmark/{bench_type}/{model_name}/{case_name}/{name}",
                        pytest_path=pytest_path,
                        task="benchmark",
                        model=model_name,
                        case=case_name,
                        extra_args=["-v", "--tb=short", "-s"],
                        extra_env={
                            "FL_BENCHMARK_TYPE": bench_type,
                            "FL_BENCHMARK_CASE": json.dumps(runtime_case),
                        },
                    )
                )
        return cases

    @staticmethod
    def _inject_model_config(runtime_case: dict[str, Any], model_cfg: ModelConfig) -> None:
        if "parameters" in runtime_case:
            runtime_case["parameters"] = model_cfg.benchmark_parameters(runtime_case.get("parameters", {}))
        if "server_parameters" in runtime_case:
            runtime_case["server_parameters"] = model_cfg.server_parameters(runtime_case.get("server_parameters", {}))
        if "client_parameters" in runtime_case:
            client_params = dict(runtime_case.get("client_parameters", {}))
            if model_cfg.serve.served_model_name:
                client_params.setdefault("model", model_cfg.serve.served_model_name)
            runtime_case["client_parameters"] = client_params

    def _run_single(self, case: TestCase) -> int:
        cmd = [sys.executable, "-m", "pytest", case.pytest_path]
        cmd.extend(case.extra_args)
        cmd.extend(self.extra_pytest_args)

        print(f"[run] --- {case.name} ---")
        if case.extra_env:
            env_text = " ".join(f"{k}={v}" for k, v in case.extra_env.items())
            print(f"[run] Env:     {env_text}")
        print(f"[run] Command: {' '.join(cmd)}")

        env = {**os.environ, **case.extra_env}
        result = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env)
        return result.returncode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, help="Platform name or device alias, e.g. cuda/a100/ascend/910b")
    parser.add_argument("--device", default=None, help="Device type within the platform, e.g. a100")
    parser.add_argument("--scope", choices=["all", "unit", "e2e", "functional", "benchmark"], default="all")
    parser.add_argument("--task", default=None, help="Task filter, e.g. inference/serving")
    parser.add_argument("--model", default=None, help="Model family filter, e.g. qwen3")
    parser.add_argument("--case", default=None, help="Model case filter, e.g. 06b_tp1")
    parser.add_argument("--benchmark", choices=["throughput", "latency", "serve"], default=None)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="Extra pytest args after '--'")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    extra_pytest_args = args.pytest_args
    if extra_pytest_args and extra_pytest_args[0] == "--":
        extra_pytest_args = extra_pytest_args[1:]

    runner = TestRunner(
        platform=args.platform,
        device=args.device,
        scope=args.scope,
        task=args.task,
        model=args.model,
        case=args.case,
        benchmark=args.benchmark,
        extra_pytest_args=extra_pytest_args,
    )
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())



