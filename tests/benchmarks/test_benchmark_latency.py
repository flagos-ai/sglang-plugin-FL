# Copyright (c) 2026 BAAI. All rights reserved.

"""Smoke test for SGLang one-batch latency benchmark."""

import sys

import pytest

from tests.benchmarks.utils import load_benchmark_case, read_last_jsonl, run_command, to_cli_args


@pytest.mark.benchmark
def test_benchmark_latency(tmp_path):
    case = load_benchmark_case()
    params = dict(case.get("parameters", {}))

    result_file = tmp_path / "latency_result.jsonl"
    command = [sys.executable, "-m", "sglang.bench_one_batch"]
    command.extend(to_cli_args(params))
    command.extend(["--result-filename", str(result_file)])

    result = run_command(command)
    assert result.returncode == 0, result.stderr
    assert result_file.exists()

    data = read_last_jsonl(result_file)
    assert data.get("prefill_latency", 0) > 0
    assert data.get("total_latency", 0) > 0
    assert data.get("overall_throughput", 0) > 0
