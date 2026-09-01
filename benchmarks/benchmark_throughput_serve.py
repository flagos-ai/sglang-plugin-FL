#!/usr/bin/env python3

# Usage:
#  1. Start the sglang server as follows (adjust model path and args as needed):
# python3 -m sglang.launch_server --model-path /models/Qwen3.6-35B-A3B --host 127.0.0.1 --port 30000 --tp 2

#  2. Run this benchmark script:
# python benchmarks/benchmark_throughput_serve.py --model /models/Qwen3.6-35B-A3B


import argparse
import csv
import os
import re
import subprocess
import time
from datetime import datetime
from statistics import mean

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 30000
DEFAULT_MODEL_NAME = "qwen"
DEFAULT_TOKENIZER_PATH = "/models/Qwen3.6-35B-A3B"

RUNS = 4

SKIP_FIRST = 1

TEST_CASES = [
    (1024, 1024, 64, 64),
    (4096, 1024, 64, 64),
    (16384, 1024, 64, 64),
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_TOKENIZER_PATH,
        help="Path to the model (used for --model and --tokenizer of sglang.bench_serving).",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Short model name used in the output directory.",
    )

    parser.add_argument(
        "--host",
        type=str,
        default=SERVER_HOST,
        help="Server host.",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=SERVER_PORT,
        help="Server port.",
    )

    return parser.parse_args()


def build_common_args(host, port, tokenizer_path):
    return [
        "python3",
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        tokenizer_path,
        "--tokenizer",
        tokenizer_path,
        "--dataset-name",
        "random-ids",
    ]


PATTERNS = {
    "successful_requests": r"Successful requests:\s+([0-9.]+)",
    "failed_requests": r"Failed requests:\s+([0-9.]+)",
    "benchmark_duration": r"Benchmark duration \(s\):\s+([0-9.]+)",
    "total_input_tokens": r"Total input tokens:\s+([0-9.]+)",
    "total_output_tokens": r"Total generated tokens:\s+([0-9.]+)",
    "request_throughput": r"Request throughput \(req/s\):\s+([0-9.]+)",
    "output_throughput": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
    "total_token_throughput": r"Total token throughput \(tok/s\):\s+([0-9.]+)",
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([0-9.]+)",
    "median_ttft_ms": r"Median TTFT \(ms\):\s+([0-9.]+)",
    "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([0-9.]+)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([0-9.]+)",
    "median_tpot_ms": r"Median TPOT \(ms\):\s+([0-9.]+)",
    "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([0-9.]+)",
    "mean_itl_ms": r"Mean ITL \(ms\):\s+([0-9.]+)",
    "median_itl_ms": r"Median ITL \(ms\):\s+([0-9.]+)",
    "p99_itl_ms": r"P99 ITL \(ms\):\s+([0-9.]+)",
    "mean_e2el_ms": r"Mean E2EL \(ms\):\s+([0-9.]+)",
    "median_e2el_ms": r"Median E2EL \(ms\):\s+([0-9.]+)",
    "p99_e2el_ms": r"P99 E2EL \(ms\):\s+([0-9.]+)",
}

RAW_CSV_COLUMNS = [
    "Prefill",
    "Decode",
    "Conc",
    "Num Prompts",
    "Successful Requests",
    "Failed Requests",
    "Run Status",
    "Benchmark Duration (s)",
    "Total Input Tokens",
    "Total Output Tokens",
    "Req/s",
    "Output tok/s",
    "Total tok/s",
    "Mean TTFT (ms)",
    "Median TTFT (ms)",
    "P99 TTFT (ms)",
    "Mean TPOT (ms)",
    "Median TPOT (ms)",
    "P99 TPOT (ms)",
    "Mean ITL (ms)",
    "Median ITL (ms)",
    "P99 ITL (ms)",
    "Mean E2EL (ms)",
    "Median E2EL (ms)",
    "P99 E2EL (ms)",
]

SUMMARY_CSV_COLUMNS = [
    "Prefill",
    "Decode",
    "Conc",
    "Num Prompts",
    "Benchmark Duration (s)",
    "Total Input Tokens",
    "Total Output Tokens",
    "Req/s",
    "Output tok/s",
    "Total tok/s",
    "Mean TTFT (ms)",
    "Median TTFT (ms)",
    "P99 TTFT (ms)",
    "Mean TPOT (ms)",
    "Median TPOT (ms)",
    "P99 TPOT (ms)",
    "Mean ITL (ms)",
    "Median ITL (ms)",
    "P99 ITL (ms)",
    "Mean E2EL (ms)",
    "Median E2EL (ms)",
    "P99 E2EL (ms)",
]


def extract_metrics(output_text):
    result = {}

    for key, pattern in PATTERNS.items():
        match = re.search(
            pattern,
            output_text,
            re.IGNORECASE,
        )

        result[key] = float(match.group(1)) if match else None

    return result


def format_result(case, metrics, include_successful_requests=True):
    input_len, output_len, concurrency, num_prompts = case

    result = {
        "Prefill": input_len,
        "Decode": output_len,
        "Conc": concurrency,
        "Num Prompts": num_prompts,
        "Benchmark Duration (s)": metrics.get("benchmark_duration"),
        "Total Input Tokens": metrics.get("total_input_tokens"),
        "Total Output Tokens": metrics.get("total_output_tokens"),
        "Req/s": metrics.get("request_throughput"),
        "Output tok/s": metrics.get("output_throughput"),
        "Total tok/s": metrics.get("total_token_throughput"),
        "Mean TTFT (ms)": metrics.get("mean_ttft_ms"),
        "Median TTFT (ms)": metrics.get("median_ttft_ms"),
        "P99 TTFT (ms)": metrics.get("p99_ttft_ms"),
        "Mean TPOT (ms)": metrics.get("mean_tpot_ms"),
        "Median TPOT (ms)": metrics.get("median_tpot_ms"),
        "P99 TPOT (ms)": metrics.get("p99_tpot_ms"),
        "Mean ITL (ms)": metrics.get("mean_itl_ms"),
        "Median ITL (ms)": metrics.get("median_itl_ms"),
        "P99 ITL (ms)": metrics.get("p99_itl_ms"),
        "Mean E2EL (ms)": metrics.get("mean_e2el_ms"),
        "Median E2EL (ms)": metrics.get("median_e2el_ms"),
        "P99 E2EL (ms)": metrics.get("p99_e2el_ms"),
    }

    if include_successful_requests:
        result["Successful Requests"] = metrics.get("successful_requests")
        result["Failed Requests"] = metrics.get("failed_requests")

    return result


def append_csv(row, filename, columns):
    file_exists = os.path.exists(filename)

    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def run_once(case, run_id, common_args):
    input_len, output_len, concurrency, num_prompts = case

    name = f"{input_len}_{output_len}_c{concurrency}"

    print("=" * 80)
    print(f"Running: {name} | Run {run_id}/{RUNS}")
    print("=" * 80)

    cmd = common_args + [
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--max-concurrency",
        str(concurrency),
        "--num-prompts",
        str(num_prompts),
    ]

    print(" ".join(cmd))
    print()

    start_time = time.time()

    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    elapsed = time.time() - start_time

    output = process.stdout

    print(output)

    metrics = extract_metrics(output)

    metrics["elapsed_sec"] = round(elapsed, 2)

    return metrics


def average_metrics(results):
    avg_result = {}

    keys = results[0].keys()

    for key in keys:
        values = [r[key] for r in results if isinstance(r.get(key), (int, float))]

        if values:
            avg_result[key] = round(mean(values), 2)

    return avg_result


def run_test_case(case, raw_csv, common_args):
    all_runs = []

    for run_id in range(1, RUNS + 1):
        metrics = run_once(case, run_id, common_args)

        raw_row = format_result(case, metrics, include_successful_requests=True)
        expected_successful_requests = case[3]
        raw_row["Run Status"] = (
            "SUCCESS"
            if metrics.get("successful_requests") == expected_successful_requests
            else "FAILED"
        )

        append_csv(raw_row, raw_csv, RAW_CSV_COLUMNS)

        all_runs.append(metrics)

    valid_runs = all_runs[SKIP_FIRST:]

    expected_successful_requests = case[3]
    has_failed_run = any(
        run.get("successful_requests") != expected_successful_requests
        for run in valid_runs
    )

    avg_metrics = average_metrics(valid_runs)

    summary_row = format_result(
        case,
        avg_metrics,
        include_successful_requests=False,
    )

    return summary_row, has_failed_run


def print_summary(results):
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    for r in results:
        print(
            f"Prefill={r['Prefill']} "
            f"Decode={r['Decode']} "
            f"Conc={r['Conc']} "
            f"NumPrompts={r['Num Prompts']} "
            f"Req/s={r['Req/s']} "
            f"Total tok/s={r['Total tok/s']} "
            f"TTFT={r['Mean TTFT (ms)']}ms "
            f"E2EL={r['Mean E2EL (ms)']}ms"
        )


def main():
    args = parse_args()

    common_args = build_common_args(args.host, args.port, args.model)

    test_cases = TEST_CASES

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = f"benchmark_results_{args.model_name}"
    os.makedirs(output_dir, exist_ok=True)
    raw_csv = os.path.join(output_dir, f"raw_runs_{timestamp}.csv")
    summary_csv = os.path.join(output_dir, f"summary_{timestamp}.csv")

    all_summary = []

    print()
    print(f"MODEL={args.model}")
    print(f"MODEL_NAME={args.model_name}")
    print(f"SERVER={args.host}:{args.port}")
    print(f"RUNS={RUNS}")
    print(f"SKIP_FIRST={SKIP_FIRST}")
    print(f"TOTAL_CASES={len(test_cases)}")
    print(f"TEST_CASES={test_cases}")
    print()

    for case in test_cases:
        try:
            summary_row, has_failed_run = run_test_case(
                case,
                raw_csv,
                common_args,
            )

            if has_failed_run:
                print(f"SKIP SUMMARY ROW (failed case): {case}")
                continue

            append_csv(summary_row, summary_csv, SUMMARY_CSV_COLUMNS)

            all_summary.append(summary_row)

        except Exception as e:
            print(f"ERROR: {e}")

    print_summary(all_summary)

    print()
    print(f"Raw CSV: {raw_csv}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
