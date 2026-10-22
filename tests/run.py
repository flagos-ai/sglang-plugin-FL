#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.

"""Unified test entry point for sglang-plugin-FL.

- tests/platforms/<platform>.yaml selects scopes/cases per device.
- tests/models/<model>/<case>.yaml stores model and engine settings.
- tests/benchmarks/configs/smoke.yaml stores benchmark smoke templates.

Examples:
    python tests/run.py --platform nvidia --device a100 --scope unit
    python tests/run.py --platform nvidia --device a100 --scope functional
    python tests/run.py --platform nvidia --device a100 --scope benchmark
    python tests/run.py --platform nvidia --device a100 --scope functional
    python tests/run.py --platform nvidia --device a100 --scope benchmark --benchmark latency
    python tests/run.py --platform nvidia --device a100 --scope functional
    python tests/run.py --platform nvidia --device a100 --scope benchmark --model qwen3 --case 06b_tp1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DIR = _REPO_ROOT / "tests"
sys.path.insert(0, str(_REPO_ROOT))

from tests.utils.device_scheduler import DeviceScheduler
from tests.utils.device_utils import get_device_count, get_visible_device_env_var
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
    engine_overrides: dict[str, Any] = field(default_factory=dict)
    extra_args: list[str] = field(default_factory=list)
    extra_env: dict[str, str] = field(default_factory=dict)
    # Total devices this case occupies (tp_size * pp_size from the model
    # config, engine overrides applied). 0 = does not use the DeviceScheduler
    # (unit, functional, benchmark run serially); >0 = the scheduler hands
    # out this many contiguous slots and runs the case concurrently with
    # other e2e cases (ported from vllm-plugin-FL #581).
    num_devices: int = 0


class TestRunner:
    def __init__(
        self,
        platform: str,
        device: str | None,
        scope: str,
        stage: str = "nightly",
        task: str | None = None,
        model: str | None = None,
        case: str | None = None,
        cases: list[dict] | None = None,
        benchmark: str | None = None,
        extra_pytest_args: list[str] | None = None,
    ) -> None:
        self.config = PlatformConfig.load(platform, device)
        self.scope = scope
        self.stage = stage
        self.task = task
        self.model = model
        self.case = case
        self.cases = cases  # explicit [{model, case}] allow-list from CI matrix
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
        print(f"[run] Stage:    {self.stage}")
        print(f"[run] Cases:    {len(cases)}")
        print()

        # Split cases: e2e cases go through the device scheduler (concurrent,
        # disjoint card groups); everything else (unit, functional, benchmark)
        # runs serially as before.
        e2e_cases = [c for c in cases if c.num_devices > 0]
        serial_cases = [c for c in cases if c.num_devices == 0]

        failed = 0
        for case in serial_cases:
            if self._run_single(case) != 0:
                failed += 1
        if e2e_cases:
            failed += self._run_e2e_parallel(e2e_cases)

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
        return [
            TestCase(
                name="unit",
                pytest_path="tests/unit_tests",
                task="unit",
                extra_args=extra_args,
            )
        ]

    def _discover_e2e_tests(self) -> list[TestCase]:
        cases: list[TestCase] = []

        # CI matrix smart-skip: when an explicit allow-list of {model, case} is
        # provided (from generate_matrix --changed-files narrowing), run only
        # those combinations.
        allowed: set[tuple[str, str]] | None = None
        if self.cases is not None:
            allowed = {(c["model"], c["case"]) for c in self.cases}

        # Future-compatible structured e2e discovery.
        for item in self.config.get_e2e_tests().get_cases(
            stage=self.stage, task=self.task, model=self.model
        ):
            model_name = item["model"]
            case_name = item["case"]
            if self.case and self.case != case_name:
                continue
            if allowed is not None and (model_name, case_name) not in allowed:
                continue
            if self.config.should_skip_model(model_name):
                print(
                    f"[run] Skipping e2e/{model_name}/{case_name} (unsupported feature)"
                )
                continue
            task = item["task"]
            test_path = f"tests/e2e_tests/{task}/test_{task}_smoke.py"
            if not (_REPO_ROOT / test_path).exists():
                print(f"[run] Warning: structured e2e test not found: {test_path}")
                continue

            engine_overrides = self.config.get_engine_overrides(model_name, case_name)

            # Device footprint for the scheduler: tp_size * pp_size, with
            # platform engine overrides applied (they may change tp). Best
            # effort only — a malformed or missing model YAML falls back to
            # 1 device rather than dropping the case.
            num_devices = 1
            try:
                model_cfg = ModelConfig.load(model_name, case_name)
                model_cfg.apply_engine_overrides(engine_overrides)
                tp = int(model_cfg.engine.get("tp_size", 1))
                pp = int(model_cfg.engine.get("pp_size", 1))
                num_devices = max(1, tp * pp)
            except Exception:
                pass

            cases.append(
                TestCase(
                    name=f"e2e/{task}/{model_name}/{case_name}",
                    pytest_path=test_path,
                    task="e2e",
                    model=model_name,
                    case=case_name,
                    engine_overrides=engine_overrides,
                    extra_args=["-v", "--tb=short", "-s"],
                    extra_env={"FL_TEST_MODEL": model_name, "FL_TEST_CASE": case_name},
                    num_devices=num_devices,
                )
            )
        return cases

    def _discover_functional_tests(self) -> list[TestCase]:
        functional_filter = self.config.get_functional_filter()
        extra_args = ["-v", "--tb=short", "-s"]
        for pattern in functional_filter.exclude:
            extra_args.extend(["--ignore", f"tests/functional_tests/{pattern}"])
        if functional_filter.include != "*" and isinstance(
            functional_filter.include, list
        ):
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
        benchmark_cfg = self.config.get_benchmark_tests(self.stage)
        if not benchmark_cfg.get("enabled", False):
            return []

        selected = benchmark_cfg.get("types", [])
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
                    print(
                        f"[run] Skipping benchmark/{model_name}/{case_name} (unsupported feature)"
                    )
                    continue

                engine_overrides = self.config.get_engine_overrides(
                    model_name,
                    case_name,
                )
                model_cfg = ModelConfig.load(model_name, case_name)
                model_cfg.apply_engine_overrides(engine_overrides)

                self._inject_model_config(runtime_case, model_cfg)
                name = str(runtime_case.get("name", f"{bench_type}_smoke"))
                cases.append(
                    TestCase(
                        name=f"benchmark/{bench_type}/{model_name}/{case_name}/{name}",
                        pytest_path=pytest_path,
                        task="benchmark",
                        model=model_name,
                        case=case_name,
                        engine_overrides=engine_overrides,
                        extra_args=["-v", "--tb=short", "-s"],
                        extra_env={
                            "FL_BENCHMARK_TYPE": bench_type,
                            "FL_BENCHMARK_CASE": json.dumps(runtime_case),
                        },
                    )
                )
        return cases

    @staticmethod
    def _inject_model_config(
        runtime_case: dict[str, Any], model_cfg: ModelConfig
    ) -> None:
        if "parameters" in runtime_case:
            runtime_case["parameters"] = model_cfg.benchmark_parameters(
                runtime_case.get("parameters", {})
            )
        if "server_parameters" in runtime_case:
            runtime_case["server_parameters"] = model_cfg.server_parameters(
                runtime_case.get("server_parameters", {})
            )
        if "client_parameters" in runtime_case:
            client_params = dict(runtime_case.get("client_parameters", {}))
            if model_cfg.serve.served_model_name:
                client_params.setdefault("model", model_cfg.serve.served_model_name)
            runtime_case["client_parameters"] = client_params

    def _run_e2e_parallel(self, cases: list[TestCase]) -> int:
        """Run e2e cases concurrently on disjoint device groups.

        A DeviceScheduler hands out non-overlapping contiguous card ranges;
        each case's subprocess gets the platform's visibility env var pinned
        to its own slots. Ported from vllm-plugin-FL #581.

        Degree of parallelism follows the case footprints:
          total=8, tp=1/2 → up to 4 cases at once
          total=8, tp=4   → up to 2 cases at once
          total=8, tp=8   → 1 case at a time (degenerates to serial)

        Falls back to serial execution (inherited visibility, no pinning)
        when the platform declares ``e2e_serial: true``, no accelerators can
        be counted, or a visibility mask already pins a subset of the cards.
        """
        total_devices = get_device_count()
        visible_env_var = get_visible_device_env_var()

        def _run_serial() -> int:
            return sum(1 for c in cases if self._run_single(c) != 0)

        if self.config.e2e_serial:
            print(
                f"[run] E2E parallel: platform '{self.config.platform}' requests "
                "serial execution (e2e_serial: true); running serially"
            )
            return _run_serial()

        if total_devices == 0:
            print(
                f"::warning::No accelerators counted for platform "
                f"'{self.config.platform}' (vendor device layer not wired into "
                "get_device_count?); e2e cases run serially without device pinning."
            )
            return _run_serial()

        # A visibility mask naming a subset of the physical cards (a locally
        # pinned run, or a host partition policy expressed as env) means the
        # counted indices are subset-relative: re-pinning with slot indices
        # could be reinterpreted against physical cards. Keep the inherited
        # visibility and run serially instead.
        mask = os.environ.get(visible_env_var, "").strip()
        if mask and mask.lower() != "all":
            try:
                masked = {int(x) for x in mask.split(",") if x.strip()}
            except ValueError:
                masked = None
            if masked is not None and masked != set(range(total_devices)):
                print(
                    f"[run] E2E parallel: {visible_env_var}={mask} pins a subset "
                    f"of the {total_devices} counted devices; keeping inherited "
                    "visibility and running serially"
                )
                return _run_serial()

        scheduler = DeviceScheduler(total_devices)
        results: dict[str, int] = {}
        results_lock = threading.Lock()

        max_concurrent = max(total_devices // max(c.num_devices, 1) for c in cases)
        print()
        print("[run] " + "=" * 60)
        print("[run] E2E parallel schedule")
        print(f"[run]   Total devices : {total_devices}  ({visible_env_var})")
        print(f"[run]   Cases         : {len(cases)}")
        print(f"[run]   Max concurrent: {max_concurrent}  (driven by smallest tp)")
        print("[run]   " + "-" * 48)
        for tc in cases:
            print(f"[run]   {tc.name:<40} {tc.num_devices:>7} device(s)")
        print("[run] " + "=" * 60)
        print()

        def _ts() -> str:
            return time.strftime("%H:%M:%S")

        def _run_one(tc: TestCase) -> None:
            slots = scheduler.acquire(tc.num_devices)
            visible = ",".join(str(s) for s in slots)
            print(
                f"[run] [{_ts()}] START  {tc.name}  {visible_env_var}={visible}  "
                f"devices={tc.num_devices}, stdout buffered until done",
                flush=True,
            )
            status = "FAIL"
            rc = 1
            try:
                device_env = {visible_env_var: visible}
                # Co-scheduled NPU-family engines each bind HCCL's default
                # port (16666); the second bind dies with a port-in-use
                # error. Give every case a disjoint port range derived from
                # its (exclusive) first slot. Stacks that ignore this
                # variable are unaffected.
                port_base = 18000 + slots[0] * 20
                device_env["HCCL_NPU_SOCKET_PORT_RANGE"] = (
                    f"{port_base}-{port_base + 19}"
                )
                # Serving smoke tests probe-then-bind an HTTP port; two
                # co-scheduled serving cases scanning from the same base can
                # both pick the same port before either server binds it.
                # Give each case a disjoint scan range derived from its
                # (exclusive) first slot.
                device_env["FL_TEST_PORT_BASE"] = str(30000 + slots[0] * 50)
                rc = self._run_single(tc, extra_env_override=device_env, quiet=True)
                status = "PASS" if rc == 0 else "FAIL"
            except Exception as exc:  # noqa: BLE001 — one case must not kill the run
                print(f"[run] [{_ts()}] ERROR  {tc.name}: {exc}", flush=True)
            finally:
                scheduler.release(slots)
                print(
                    f"[run] [{_ts()}] {status:<6} {tc.name}  "
                    f"released {visible_env_var}={visible}",
                    flush=True,
                )
            with results_lock:
                results[tc.name] = rc

        # Worst case all tp=1: one worker per device.
        with ThreadPoolExecutor(max_workers=total_devices) as pool:
            futures = [pool.submit(_run_one, tc) for tc in cases]
            for fut in as_completed(futures):
                fut.result()  # propagate worker exceptions

        return sum(1 for tc in cases if results.get(tc.name, 1) != 0)

    def _run_single(
        self,
        case: TestCase,
        extra_env_override: dict[str, str] | None = None,
        quiet: bool = False,
    ) -> int:
        """Run one test case via a pytest subprocess.

        Args:
            case: The test case descriptor.
            extra_env_override: Env vars merged last (highest priority); the
                parallel e2e path injects VISIBLE_DEVICES pinning here.
            quiet: Skip the header lines and capture subprocess output (the
                parallel path already emits a timestamped START line and
                replays captured output with a per-case prefix).
        """
        cmd = [sys.executable, "-m", "pytest", case.pytest_path]
        cmd.extend(case.extra_args)
        cmd.extend(self.extra_pytest_args)

        if not quiet:
            print(f"[run] --- {case.name} ---")
            if case.extra_env:
                env_text = " ".join(f"{k}={v}" for k, v in case.extra_env.items())
                print(f"[run] Env:     {env_text}")
            if extra_env_override:
                dev_str = " ".join(f"{k}={v}" for k, v in extra_env_override.items())
                print(f"[run] Devices: {dev_str}")
            print(f"[run] Command: {' '.join(cmd)}")

        extra_env = dict(case.extra_env)
        if case.task == "e2e" and case.model and case.case:
            extra_env["FL_TEST_ENGINE_OVERRIDES"] = json.dumps(case.engine_overrides)

        env = {**os.environ, **extra_env}
        if extra_env_override:
            env.update(extra_env_override)

        result = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            env=env,
            capture_output=quiet,
            text=quiet,
        )
        if quiet and result.stdout:
            for line in result.stdout.splitlines():
                print(f"[{case.name}] {line}")
        if quiet and result.stderr:
            for line in result.stderr.splitlines():
                print(f"[{case.name}] {line}")
        return result.returncode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        required=True,
        help="Platform name or device alias, e.g. nvidia/a100/ascend/910b",
    )
    parser.add_argument(
        "--device", default=None, help="Device type within the platform, e.g. a100"
    )
    parser.add_argument(
        "--scope",
        choices=["all", "unit", "e2e", "functional", "benchmark"],
        default="all",
    )
    parser.add_argument(
        "--stage",
        choices=["pr", "nightly", "weekly"],
        default="nightly",
        help="CI stage for e2e/benchmark case selection: pr (subset), nightly "
        "(full), weekly (falls back to nightly if not defined) (default: nightly)",
    )
    parser.add_argument(
        "--task", default=None, help="Task filter, e.g. inference/serving"
    )
    parser.add_argument("--model", default=None, help="Model family filter, e.g. qwen3")
    parser.add_argument("--case", default=None, help="Model case filter, e.g. 06b_tp1")
    parser.add_argument(
        "--cases",
        default=None,
        help="JSON array of {model,case} dicts to run (CI matrix smart-skip).",
    )
    parser.add_argument(
        "--benchmark", choices=["throughput", "latency", "serve"], default=None
    )
    parser.add_argument(
        "pytest_args", nargs=argparse.REMAINDER, help="Extra pytest args after '--'"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    extra_pytest_args = args.pytest_args
    if extra_pytest_args and extra_pytest_args[0] == "--":
        extra_pytest_args = extra_pytest_args[1:]

    cases_filter: list[dict] | None = None
    if args.cases:
        cases_filter = json.loads(args.cases)

    runner = TestRunner(
        platform=args.platform,
        device=args.device,
        scope=args.scope,
        stage=args.stage,
        task=args.task,
        model=args.model,
        case=args.case,
        cases=cases_filter,
        benchmark=args.benchmark,
        extra_pytest_args=extra_pytest_args,
    )
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
