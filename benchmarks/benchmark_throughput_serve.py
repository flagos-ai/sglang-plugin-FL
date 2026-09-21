#!/usr/bin/env python3
"""Run the fixed-shape Qwen3.6 online-serving acceptance benchmark.

The server must already be listening. Each shape is run four times; the first
run is retained in the raw artifacts but excluded from the summary average.
Every measured request must complete with the exact requested token lengths.

Example:
  python benchmarks/benchmark_throughput_serve.py \
    --model /models/Qwen3.6-35B-A3B \
    --model-name qwen3.6-35b-a3b \
    --host 127.0.0.1 --port 30000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


SERVER_HOST = "127.0.0.1"
SERVER_PORT = 30000
DEFAULT_MODEL_NAME = "qwen3.6-35b-a3b"
DEFAULT_TOKENIZER_PATH = "/models/Qwen3.6-35B-A3B"

RUNS = 4
SKIP_FIRST = 1
TEST_CASES = [
    (1024, 1024, 64, 64),
    (4096, 1024, 64, 64),
    (16384, 1024, 64, 64),
]


JSON_METRICS = {
    "successful_requests": "completed",
    "benchmark_duration": "duration",
    "total_input_tokens": "total_input_tokens",
    "total_output_tokens": "total_output_tokens",
    "request_throughput": "request_throughput",
    "input_throughput": "input_throughput",
    "output_throughput": "output_throughput",
    "total_token_throughput": "total_throughput",
    "concurrency": "concurrency",
    "peak_output_throughput": "max_output_tokens_per_s",
    "peak_concurrent_requests": "max_concurrent_requests",
    "mean_ttft_ms": "mean_ttft_ms",
    "median_ttft_ms": "median_ttft_ms",
    "p99_ttft_ms": "p99_ttft_ms",
    "mean_tpot_ms": "mean_tpot_ms",
    "median_tpot_ms": "median_tpot_ms",
    "p99_tpot_ms": "p99_tpot_ms",
    "mean_itl_ms": "mean_itl_ms",
    "median_itl_ms": "median_itl_ms",
    "p99_itl_ms": "p99_itl_ms",
    "mean_e2e_ms": "mean_e2e_latency_ms",
    "median_e2e_ms": "median_e2e_latency_ms",
    "p99_e2e_ms": "p99_e2e_latency_ms",
}

RAW_CSV_COLUMNS = [
    "Prefill",
    "Decode",
    "Conc",
    "Num Prompts",
    "Run",
    "Included in Summary",
    "Successful Requests",
    "Failed Requests",
    "Run Status",
    "Benchmark Duration (s)",
    "Wall Clock (s)",
    "Total Input Tokens",
    "Total Output Tokens",
    "Req/s",
    "Input tok/s",
    "Output tok/s",
    "Total tok/s",
    "Concurrency",
    "Peak Output tok/s",
    "Peak Concurrent Requests",
    "Mean TTFT (ms)",
    "Median TTFT (ms)",
    "P99 TTFT (ms)",
    "Mean TPOT (ms)",
    "Median TPOT (ms)",
    "P99 TPOT (ms)",
    "Mean ITL (ms)",
    "Median ITL (ms)",
    "P99 ITL (ms)",
    "Mean E2E (ms)",
    "Median E2E (ms)",
    "P99 E2E (ms)",
]

SUMMARY_CSV_COLUMNS = [
    column
    for column in RAW_CSV_COLUMNS
    if column
    not in {
        "Run",
        "Included in Summary",
        "Successful Requests",
        "Failed Requests",
        "Run Status",
    }
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_TOKENIZER_PATH,
        help="Local model/tokenizer path passed to the SGLang benchmark client.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=(
            "Model name advertised by the server; it is also used as the "
            "filesystem-safe result label."
        ),
    )
    parser.add_argument("--host", default=SERVER_HOST, help="SGLang server host.")
    parser.add_argument(
        "--port", type=int, default=SERVER_PORT, help="SGLang server port."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Artifact root. Defaults to benchmark_results_<model-name>; a "
            "timestamped run directory is created beneath it."
        ),
    )
    return parser.parse_args(argv)


def build_common_args(
    host: str, port: int, tokenizer_path: str, served_model_name: str
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        tokenizer_path,
        "--served-model-name",
        served_model_name,
        "--tokenizer",
        tokenizer_path,
        "--dataset-name",
        "random-ids",
        "--random-range-ratio",
        "1.0",
        "--tokenize-prompt",
        "--flush-cache",
        "--warmup-requests",
        "1",
        "--request-rate",
        "inf",
        "--seed",
        "42",
        "--disable-tqdm",
        "--output-details",
    ]


def _safe_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return sanitized or "model"


def _read_single_jsonl(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"SGLang did not create its JSONL result: {path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 1:
        raise RuntimeError(
            f"Expected exactly one JSONL record in {path}, found {len(lines)}"
        )
    try:
        record = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid benchmark JSONL in {path}: {exc}") from exc
    if not isinstance(record, dict):
        raise RuntimeError(f"Benchmark JSONL record is not an object: {path}")
    return record


def _require_number(record: dict[str, Any], key: str) -> int | float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Missing or non-numeric benchmark metric: {key}")
    if not math.isfinite(float(value)):
        raise RuntimeError(f"Non-finite benchmark metric {key}: {value}")
    return value


def extract_and_validate_metrics(
    record: dict[str, Any], case: tuple[int, int, int, int]
) -> dict[str, int | float]:
    input_len, output_len, _concurrency, num_prompts = case
    metrics = {
        metric_name: _require_number(record, json_key)
        for metric_name, json_key in JSON_METRICS.items()
    }

    expected_input_tokens = input_len * num_prompts
    expected_output_tokens = output_len * num_prompts
    checks = {
        "completed requests": (metrics["successful_requests"], num_prompts),
        "total input tokens": (metrics["total_input_tokens"], expected_input_tokens),
        "total output tokens": (
            metrics["total_output_tokens"],
            expected_output_tokens,
        ),
    }
    mismatches = [
        f"{name}: got {actual}, expected {expected}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]

    details = {
        "input_lens": (record.get("input_lens"), input_len),
        "output_lens": (record.get("output_lens"), output_len),
    }
    for key, (values, expected) in details.items():
        if not isinstance(values, list) or len(values) != num_prompts:
            length = len(values) if isinstance(values, list) else "N/A"
            mismatches.append(
                f"{key}: expected a {num_prompts}-item list, got "
                f"{type(values).__name__} of length {length}"
            )
        elif any(value != expected for value in values):
            mismatches.append(f"{key}: not every request has length {expected}")

    errors = record.get("errors")
    if not isinstance(errors, list) or len(errors) != num_prompts:
        mismatches.append("errors: missing per-request error details")
    else:
        request_errors = [error for error in errors if error]
        if request_errors:
            mismatches.append(
                f"errors: {len(request_errors)} request(s) failed; "
                f"first={request_errors[0]!r}"
            )

    if mismatches:
        raise RuntimeError("; ".join(mismatches))

    metrics["failed_requests"] = 0
    return metrics


def format_result(
    case: tuple[int, int, int, int],
    metrics: dict[str, int | float],
    *,
    include_status: bool,
) -> dict[str, Any]:
    input_len, output_len, concurrency, num_prompts = case
    result: dict[str, Any] = {
        "Prefill": input_len,
        "Decode": output_len,
        "Conc": concurrency,
        "Num Prompts": num_prompts,
        "Benchmark Duration (s)": metrics.get("benchmark_duration"),
        "Wall Clock (s)": metrics.get("elapsed_sec"),
        "Total Input Tokens": metrics.get("total_input_tokens"),
        "Total Output Tokens": metrics.get("total_output_tokens"),
        "Req/s": metrics.get("request_throughput"),
        "Input tok/s": metrics.get("input_throughput"),
        "Output tok/s": metrics.get("output_throughput"),
        "Total tok/s": metrics.get("total_token_throughput"),
        "Concurrency": metrics.get("concurrency"),
        "Peak Output tok/s": metrics.get("peak_output_throughput"),
        "Peak Concurrent Requests": metrics.get("peak_concurrent_requests"),
        "Mean TTFT (ms)": metrics.get("mean_ttft_ms"),
        "Median TTFT (ms)": metrics.get("median_ttft_ms"),
        "P99 TTFT (ms)": metrics.get("p99_ttft_ms"),
        "Mean TPOT (ms)": metrics.get("mean_tpot_ms"),
        "Median TPOT (ms)": metrics.get("median_tpot_ms"),
        "P99 TPOT (ms)": metrics.get("p99_tpot_ms"),
        "Mean ITL (ms)": metrics.get("mean_itl_ms"),
        "Median ITL (ms)": metrics.get("median_itl_ms"),
        "P99 ITL (ms)": metrics.get("p99_itl_ms"),
        "Mean E2E (ms)": metrics.get("mean_e2e_ms"),
        "Median E2E (ms)": metrics.get("median_e2e_ms"),
        "P99 E2E (ms)": metrics.get("p99_e2e_ms"),
    }
    if include_status:
        result.update(
            {
                "Successful Requests": metrics.get("successful_requests"),
                "Failed Requests": metrics.get("failed_requests"),
                "Run Status": "SUCCESS",
            }
        )
    return result


def append_csv(row: dict[str, Any], filename: Path, columns: list[str]) -> None:
    file_exists = filename.exists()
    with filename.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_once(
    case: tuple[int, int, int, int],
    run_id: int,
    common_args: list[str],
    artifact_dir: Path,
) -> dict[str, int | float]:
    input_len, output_len, concurrency, num_prompts = case
    name = f"{input_len}_{output_len}_c{concurrency}_run{run_id}"
    result_file = (artifact_dir / f"{name}.jsonl").resolve()
    stdout_file = artifact_dir / f"{name}.log"
    if result_file.exists():
        result_file.unlink()

    cmd = common_args + [
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--max-concurrency",
        str(concurrency),
        "--num-prompts",
        str(num_prompts),
        "--output-file",
        str(result_file),
    ]

    print("=" * 80)
    print(f"Running: {name} | Run {run_id}/{RUNS}")
    print("=" * 80)
    print(" ".join(cmd), flush=True)

    started = time.monotonic()
    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.monotonic() - started
    stdout_file.write_text(process.stdout, encoding="utf-8")
    print(process.stdout)
    if process.returncode != 0:
        raise RuntimeError(
            f"Benchmark subprocess failed with exit code {process.returncode}; "
            f"see {stdout_file}"
        )

    record = _read_single_jsonl(result_file)
    metrics = extract_and_validate_metrics(record, case)
    metrics["elapsed_sec"] = round(elapsed, 2)
    return metrics


def average_metrics(
    results: list[dict[str, int | float]],
) -> dict[str, int | float]:
    averaged: dict[str, int | float] = {}
    for key in results[0]:
        values = [float(result[key]) for result in results if key in result]
        if values:
            averaged[key] = round(mean(values), 2)
    return averaged


def run_test_case(
    case: tuple[int, int, int, int],
    raw_csv: Path,
    common_args: list[str],
    artifact_dir: Path,
) -> dict[str, Any]:
    all_runs: list[dict[str, int | float]] = []
    for run_id in range(1, RUNS + 1):
        metrics = run_once(case, run_id, common_args, artifact_dir)
        raw_row = format_result(case, metrics, include_status=True)
        raw_row["Run"] = run_id
        raw_row["Included in Summary"] = "NO" if run_id <= SKIP_FIRST else "YES"
        append_csv(
            raw_row,
            raw_csv,
            RAW_CSV_COLUMNS,
        )
        all_runs.append(metrics)

    valid_runs = all_runs[SKIP_FIRST:]
    if len(valid_runs) != RUNS - SKIP_FIRST:
        raise RuntimeError(
            f"Expected {RUNS - SKIP_FIRST} measured runs, got {len(valid_runs)}"
        )
    return format_result(
        case,
        average_metrics(valid_runs),
        include_status=False,
    )


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("Summary (mean of runs 2-4)")
    print("=" * 80)
    for result in results:
        print(
            f"Prefill={result['Prefill']} Decode={result['Decode']} "
            f"Conc={result['Conc']} NumPrompts={result['Num Prompts']} "
            f"Req/s={result['Req/s']} Total tok/s={result['Total tok/s']} "
            f"TTFT={result['Mean TTFT (ms)']}ms "
            f"E2E={result['Mean E2E (ms)']}ms"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    common_args = build_common_args(args.host, args.port, args.model, args.model_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = (
        Path(args.output_dir)
        if args.output_dir
        else Path(f"benchmark_results_{_safe_component(args.model_name)}")
    )
    output_dir = root / timestamp
    artifact_dir = output_dir / "official-jsonl"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    raw_csv = output_dir / "raw_runs.csv"
    summary_csv = output_dir / "summary.csv"

    configuration = {
        "model": args.model,
        "model_name": args.model_name,
        "server": f"{args.host}:{args.port}",
        "runs": RUNS,
        "skip_first": SKIP_FIRST,
        "test_cases": TEST_CASES,
        "client_common_args": common_args,
    }
    (output_dir / "configuration.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(configuration, indent=2))
    print(f"Artifacts: {output_dir.resolve()}\n")

    summaries: list[dict[str, Any]] = []
    failures: list[str] = []
    for case in TEST_CASES:
        try:
            summary = run_test_case(case, raw_csv, common_args, artifact_dir)
        except Exception as exc:
            message = f"case={case}: {exc}"
            failures.append(message)
            print(f"ERROR: {message}", file=sys.stderr, flush=True)
            continue
        append_csv(summary, summary_csv, SUMMARY_CSV_COLUMNS)
        summaries.append(summary)

    print_summary(summaries)
    print(f"\nRaw CSV: {raw_csv.resolve()}")
    print(f"Summary CSV: {summary_csv.resolve()}")

    if failures or len(summaries) != len(TEST_CASES):
        failure_file = output_dir / "failures.txt"
        failure_file.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"FAILED: {len(failures)} case(s); see {failure_file.resolve()}")
        return 1

    print("PASS: all 12 runs completed with exact request and token counts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
