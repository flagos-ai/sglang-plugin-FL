# Copyright (c) 2026 BAAI. All rights reserved.

"""Contract tests for the fixed-shape Ascend serving benchmark driver."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_DRIVER_PATH = (
    Path(__file__).parents[2] / "benchmarks" / "benchmark_throughput_serve.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "ascend_benchmark_throughput_serve", _DRIVER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_DRIVER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _DRIVER
_SPEC.loader.exec_module(_DRIVER)


def _record(input_len: int = 1024, output_len: int = 1024) -> dict:
    num_prompts = 64
    record = {json_key: 1.0 for json_key in _DRIVER.JSON_METRICS.values()}
    record.update(
        {
            "completed": num_prompts,
            "total_input_tokens": input_len * num_prompts,
            "total_output_tokens": output_len * num_prompts,
            "input_lens": [input_len] * num_prompts,
            "output_lens": [output_len] * num_prompts,
            "errors": [None] * num_prompts,
        }
    )
    return record


def test_client_uses_the_name_advertised_by_the_server() -> None:
    args = _DRIVER.build_common_args(
        "127.0.0.1", 30000, "/models/Qwen3.6-27B", "qwen3_6_27b"
    )

    model_name_index = args.index("--served-model-name")
    assert args[model_name_index + 1] == "qwen3_6_27b"
    assert "random-ids" in args
    assert "--tokenize-prompt" in args


def test_exact_request_and_token_counts_pass() -> None:
    metrics = _DRIVER.extract_and_validate_metrics(_record(), (1024, 1024, 64, 64))

    assert metrics["successful_requests"] == 64
    assert metrics["failed_requests"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("completed", 63),
        ("total_input_tokens", 1024 * 64 - 1),
        ("total_output_tokens", 1024 * 64 - 1),
        ("output_lens", [1023] * 64),
        ("errors", ["request failed"] + [None] * 63),
    ],
)
def test_partial_or_inexact_run_fails(field: str, value: object) -> None:
    record = _record()
    record[field] = value

    with pytest.raises(RuntimeError):
        _DRIVER.extract_and_validate_metrics(record, (1024, 1024, 64, 64))
