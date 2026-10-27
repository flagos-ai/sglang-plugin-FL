#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Run Iluvatar's four platform stages with explicit, non-empty CI evidence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from xml.etree import ElementTree

REPO_ROOT = Path(__file__).resolve().parents[3]
UNIT_TESTS = (
    "tests/unit_tests/platform/test_iluvatar_defaults.py",
    "tests/unit_tests/platform/test_iluvatar_import_shims.py",
    "tests/unit_tests/platform/test_iluvatar_runtime.py",
    "tests/unit_tests/dispatch/backends/test_iluvatar_fla.py",
    "tests/unit_tests/dispatch/backends/test_iluvatar_native_contracts.py",
    "tests/unit_tests/dispatch/backends/test_flagos_empty_ops.py",
    "tests/unit_tests/dispatch/bridge/test_fla_ops.py",
    "tests/unit_tests/test_iluvatar_example_graphs.py",
    "tests/unit_tests/test_iluvatar_example_sampling.py",
    "tests/unit_tests/test_iluvatar_test_config.py",
    "tests/unit_tests/test_iluvatar_benchmark_config.py",
)
FUNCTIONAL_TESTS = (
    "tests/functional_tests/ops/test_iluvatar_empty_ops.py",
    "tests/unit_tests/platform/test_iluvatar_runtime.py",
)
MIN_UNIT_TESTS = 148
MIN_FUNCTIONAL_TESTS = 31
BENCHMARKS = ("throughput", "latency", "serve")


def _check_results(report: Path, minimum: int) -> None:
    path = REPO_ROOT / "scripts/iluvatar/check_ci_results.py"
    spec = importlib.util.spec_from_file_location("_iluvatar_ci_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_results(report, minimum)


def _run(command: list[str], env: dict[str, str]) -> None:
    print("[iluvatar] " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=str(REPO_ROOT), env=env)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def _run_report(command, env, report, minimum):
    report.parent.mkdir(parents=True, exist_ok=True)
    # A failed or empty invocation must never reuse a previous run's evidence.
    report.unlink(missing_ok=True)
    _run([*command, f"--junitxml={report}"], env)
    _check_results(report, minimum)


def _environment() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return env


def _workload(env: dict[str, str]) -> None:
    env.setdefault("CONCURRENT_N", "16")
    env.setdefault("MAX_TOKENS", "256")
    for key, minimum in (
        ("CONCURRENT_N", 2),
        ("MAX_TOKENS", 1),
        ("FL_CONCURRENT_N", 2),
        ("FL_CONCURRENT_MAX_TOKENS", 1),
        ("FL_CONCURRENT_VL_MAX_TOKENS", 1),
    ):
        if key in env and (
            not re.fullmatch(r"[1-9][0-9]*", env[key]) or int(env[key]) < minimum
        ):
            raise ValueError(f"{key} must be an integer >= {minimum}")
    # Keep the original examples and shared concurrent VL workload at 256 by default.
    env.setdefault("FL_CONCURRENT_MAX_TOKENS", env["MAX_TOKENS"])
    env.setdefault("FL_CONCURRENT_VL_MAX_TOKENS", "256")


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", value
    ):
        raise ValueError(f"Invalid model/case identifier: {value!r}")
    return value


def _e2e_cases(args):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tests.run import TestRunner

    requested = None
    if args.cases is not None:
        if not args.cases.strip():
            raise ValueError("Explicit e2e --cases must not be empty")
        requested = json.loads(args.cases)
        if not isinstance(requested, list) or not requested:
            raise ValueError("Explicit e2e --cases must be a non-empty JSON list")
        pairs = []
        for item in requested:
            if not isinstance(item, dict) or not {"model", "case"} <= item.keys():
                raise ValueError("Each e2e case requires model and case")
            pairs.append((_identifier(item["model"]), _identifier(item["case"])))
        if len(set(pairs)) != len(pairs):
            raise ValueError("Duplicate e2e cases would overwrite their reports")

    runner = TestRunner(
        "iluvatar",
        args.device,
        "e2e",
        task=args.task,
        model=args.model,
        case=args.case,
        cases=requested,
    )
    selected = runner.discover_tests()
    if not selected:
        raise ValueError("Iluvatar e2e selection must contain at least one case")
    result = []
    for case in selected:
        task = case.name.split("/")[1]
        if task not in ("inference", "concurrent", "serving"):
            raise ValueError(f"Unsupported Iluvatar e2e task: {task}")
        result.append((task, _identifier(case.model), _identifier(case.case)))
    if requested is not None:
        found = {(model, case) for _, model, case in result}
        if set(pairs) != found:
            raise ValueError(
                "Some requested e2e cases were not selected by the platform manifest"
            )
    return result


def _accept_models(env, artifacts):
    for profile in ("vendor", "flagos"):
        output = artifacts / f"models-{profile}"
        logs = [
            output / f"{model}_{mode}_graph_disabled_{disabled}.log"
            for model in ("27b", "35b_a3b")
            for disabled in (1, 0)
            for mode in ("offline", "concurrent")
        ]
        logs.extend((output / "environment.log", output / "flagcx.log"))
        for log in logs:
            log.unlink(missing_ok=True)
        child_env = dict(env)
        child_env.pop("SGLANG_FL_PER_OP", None)
        child_env.update(
            FUSED_PROFILE=profile,
            TP_SIZE="4",
            OUTPUT_DIR=str(output),
            SGLANG_FL_DIST_BACKEND="flagcx",
        )
        _run(["bash", "scripts/iluvatar/run_single_node_acceptance.sh"], child_env)
        for log in logs:
            if not log.is_file() or not log.stat().st_size:
                raise RuntimeError(f"Missing or empty model evidence: {log}")


def run_stage(args) -> None:
    env = _environment()
    artifacts = Path(env.get("ILUVATAR_CI_ARTIFACT_DIR", "artifacts/iluvatar"))
    if not artifacts.is_absolute():
        artifacts = REPO_ROOT / artifacts
    extra = args.pytest_args
    if extra and extra[0] == "--":
        extra = extra[1:]

    if args.scope == "unit":
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            *UNIT_TESTS,
            *extra,
            "-m",
            "not gpu",
        ]
        _run_report(command, env, artifacts / "unit/contracts.xml", MIN_UNIT_TESTS)
    elif args.scope == "functional":
        # Import SGLang before collecting tests, as in the original GPU gate.
        code = "import sglang\nimport pytest\nimport sys\nraise SystemExit(pytest.main(sys.argv[1:]))"
        command = [
            sys.executable,
            "-c",
            code,
            "-q",
            "-p",
            "no:cacheprovider",
            *FUNCTIONAL_TESTS,
            *extra,
            "-m",
            "gpu",
        ]
        _run_report(
            command, env, artifacts / "functional/contracts.xml", MIN_FUNCTIONAL_TESTS
        )
    elif args.scope == "e2e":
        _workload(env)
        selected = _e2e_cases(args)
        for task, model, case in selected:
            command = [
                sys.executable,
                "tests/run.py",
                "--platform",
                "iluvatar",
                "--device",
                args.device,
                "--scope",
                "e2e",
                "--task",
                task,
                "--model",
                model,
                "--case",
                case,
                "--",
                *extra,
            ]
            _run_report(
                command,
                env,
                artifacts / "e2e" / task / model / f"{case}.xml",
                5 if task == "serving" else 1,
            )
        if any(task == "inference" for task, _, _ in selected):
            _accept_models(env, artifacts)
    else:
        for benchmark in BENCHMARKS:
            if args.benchmark and args.benchmark != benchmark:
                continue
            command = [
                sys.executable,
                "tests/run.py",
                "--platform",
                "iluvatar",
                "--device",
                args.device,
                "--scope",
                "benchmark",
                "--benchmark",
                benchmark,
            ]
            if args.model:
                command.extend(("--model", args.model))
            if args.case:
                command.extend(("--case", args.case))
            command.extend(("--", *extra))
            _run_report(command, env, artifacts / "benchmark" / f"{benchmark}.xml", 1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("iluvatar",), required=True)
    parser.add_argument(
        "--scope", choices=("unit", "functional", "e2e", "benchmark"), required=True
    )
    parser.add_argument("--device", default="bi_v150")
    parser.add_argument("--task", choices=("inference", "concurrent", "serving"))
    parser.add_argument("--cases")
    parser.add_argument("--model")
    parser.add_argument("--case")
    parser.add_argument("--benchmark", choices=BENCHMARKS)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    args.device = args.device or "bi_v150"
    try:
        run_stage(args)
    except subprocess.CalledProcessError as error:
        print(f"[iluvatar] Stage failed: {error}", file=sys.stderr)
        return error.returncode if error.returncode > 0 else 1
    except (OSError, RuntimeError, ValueError, ElementTree.ParseError) as error:
        print(f"[iluvatar] Stage failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
