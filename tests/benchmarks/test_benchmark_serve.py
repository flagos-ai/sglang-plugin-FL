# Copyright (c) 2026 BAAI. All rights reserved.

"""Smoke test for SGLang serving benchmark."""

import sys

import pytest

from tests.benchmarks.utils import (
    load_benchmark_case,
    read_last_jsonl,
    run_command,
    to_cli_args,
)
from tests.e2e_tests.serving.server_helper import SGLangServer


@pytest.mark.benchmark
def test_benchmark_serve(tmp_path):
    case = load_benchmark_case()
    
    server_params = dict(case.get("server_parameters", {}))
    client_params = dict(case.get("client_parameters", {}))

    model_path = server_params.pop("model_path")
    served_model_name = server_params.pop("served_model_name", "smoke-model")
    client_params.setdefault("model", served_model_name)
    client_params.setdefault("served_model_name", served_model_name)
    client_params.setdefault("tokenizer", model_path)
    tp_size = int(server_params.pop("tp_size", server_params.pop("tensor_parallel_size", 1)))
    host = server_params.pop("host", "127.0.0.1")
    port = int(server_params.pop("port", 0) or 0)
    server_extra_args = to_cli_args(server_params)

    result_file = tmp_path / "serve_result.jsonl"

    with SGLangServer(
        model_path=model_path,
        tp_size=tp_size,
        served_model_name=served_model_name,
        host=host,
        port=port or None,
        extra_args=server_extra_args,
    ) as server:
        command = [
            sys.executable,
            "-m",
            "sglang.bench_serving",
            "--host",
            server.host,
            "--port",
            str(server.port),
        ]
        command.extend(to_cli_args(client_params))
        command.extend(["--output-file", str(result_file), "--output-details"])

        result = run_command(command)

    assert result.returncode == 0, result.stderr
    assert result_file.exists()

    data = read_last_jsonl(result_file)
    expected = int(client_params.get("num_prompts", 0))
    assert data.get("completed", 0) > 0
    if expected:
        assert data.get("completed", 0) == expected
    errors = [err for err in data.get("errors", []) if err]
    assert not errors


