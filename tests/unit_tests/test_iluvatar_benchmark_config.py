# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Discover and launch the original benchmark smoke entrypoints on CoreX."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import run as runner_module


_ROOT = Path(__file__).resolve().parents[2]
_TYPES = ("throughput", "latency", "serve")


def _runner(**kwargs):
    return runner_module.TestRunner("iluvatar", "bi_v150", "benchmark", **kwargs)


def _runtime_cases(runner):
    return {
        case.extra_env["FL_BENCHMARK_TYPE"]: (
            case,
            json.loads(case.extra_env["FL_BENCHMARK_CASE"]),
        )
        for case in runner.discover_tests()
    }


def _track_environment(monkeypatch, config):
    # run() writes these values; retain and restore any caller's environment.
    for name, value in config.env_defaults.items():
        monkeypatch.setenv(name, str(value))
    monkeypatch.setenv("FL_TEST_PLATFORM", "iluvatar")
    monkeypatch.setenv("FL_TEST_DEVICE", "bi_v150")


def test_iluvatar_discovers_all_three_original_benchmark_entrypoints():
    runner = _runner(model="qwen3_6", case="27b_tp4_nograph")
    cases = _runtime_cases(runner)
    assert set(cases) == set(_TYPES)
    assert len(runner.discover_tests()) == 3
    for benchmark, (case, _) in cases.items():
        assert case.pytest_path == f"tests/benchmarks/test_benchmark_{benchmark}.py"
        assert case.model == "qwen3_6"
        assert case.case == "27b_tp4_nograph"


@pytest.mark.parametrize("benchmark", _TYPES)
def test_binding_preserves_shared_smoke_workload_and_inherits_corex_engine(benchmark):
    templates = runner_module.load_structured_config(
        _ROOT / "tests/benchmarks/configs/smoke.yaml"
    )
    template = templates[benchmark][0]
    case, runtime = _runtime_cases(_runner())[benchmark]
    assert runtime["name"] == template["name"]
    for section in ("parameters", "server_parameters", "client_parameters"):
        for name, expected in template.get(section, {}).items():
            if section == "server_parameters" and name == "tp_size":
                expected = 4
            assert runtime[section][name] == expected

    section = "server_parameters" if benchmark == "serve" else "parameters"
    params = runtime[section]
    assert params["model_path"] == "/models/Qwen3.6-27B"
    assert params["tp_size"] == 4
    assert params["dtype"] == "bfloat16"
    assert params["attention_backend"] == "triton"
    assert params["sampling_backend"] == "pytorch"
    assert params["mm_attention_backend"] == "triton_attn"
    assert params["disable_fast_image_processor"] is True
    assert params["disable_custom_all_reduce"] is True
    assert params["load_format"] == "dummy"
    assert params["disable_cuda_graph"] is True
    assert case.engine_overrides["model_path"] == params["model_path"]
    if benchmark == "serve":
        assert runtime["client_parameters"]["model"] == params["served_model_name"]


@pytest.mark.parametrize("benchmark", _TYPES)
def test_cli_launches_each_original_pytest_with_bound_runtime_case(
    monkeypatch, benchmark
):
    runner = _runner()
    _track_environment(monkeypatch, runner.config)
    calls = []

    def subprocess_run(command, *, cwd, env):
        calls.append((command, cwd, env))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner_module.subprocess, "run", subprocess_run)
    report = f"artifacts/benchmark-{benchmark}.xml"
    result = runner_module.main(
        [
            "--platform",
            "iluvatar",
            "--device",
            "bi_v150",
            "--scope",
            "benchmark",
            "--benchmark",
            benchmark,
            "--",
            f"--junitxml={report}",
        ]
    )

    assert result == 0
    assert len(calls) == 1
    command, cwd, env = calls[0]
    assert command[:4] == [
        runner_module.sys.executable,
        "-m",
        "pytest",
        f"tests/benchmarks/test_benchmark_{benchmark}.py",
    ]
    assert f"--junitxml={report}" in command
    assert cwd == str(_ROOT)
    assert env["FL_BENCHMARK_TYPE"] == benchmark
    runtime = json.loads(env["FL_BENCHMARK_CASE"])
    section = "server_parameters" if benchmark == "serve" else "parameters"
    assert runtime[section]["model_path"] == "/models/Qwen3.6-27B"
    assert runtime[section]["tp_size"] == 4
    assert runtime[section]["dtype"] == "bfloat16"
    assert runtime[section]["attention_backend"] == "triton"
    assert runtime[section]["sampling_backend"] == "pytorch"
    assert env["SGLANG_FL_DIST_BACKEND"] == "flagcx"
    assert env["FLAGCX_PATH"] == "/sgl-workspace/FlagCX"


def test_empty_iluvatar_benchmark_selection_fails_before_subprocess(monkeypatch):
    runner = _runner(model="qwen3", case="06b_tp1")
    _track_environment(monkeypatch, runner.config)

    def unexpected(*args, **kwargs):
        raise AssertionError("an empty selection must not launch a subprocess")

    monkeypatch.setattr(runner_module.subprocess, "run", unexpected)
    assert runner.discover_tests() == []
    assert runner.run() == 1


def test_other_platform_shared_model_binding_is_unchanged():
    runner = runner_module.TestRunner("cuda", None, "benchmark")
    cases = _runtime_cases(runner)
    assert set(cases) == set(_TYPES)
    for case, _ in cases.values():
        assert case.model == "qwen3"
        assert case.case == "06b_tp1"
    assert cases["serve"][1]["server_parameters"]["tp_size"] == 1


def test_optional_empty_selection_policy_leaves_other_platforms_unchanged(monkeypatch):
    runner = runner_module.TestRunner("cuda", None, "benchmark")
    _track_environment(monkeypatch, runner.config)
    runner.config.device_tests[runner.config.device]["tests"]["benchmark"] = {
        "enabled": False
    }
    assert runner.discover_tests() == []
    assert runner.run() == 0
