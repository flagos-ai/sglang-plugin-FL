# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Profile the target-platform SGLang baseline with concurrent requests.

The warmup and profiled batches are identical.  The warmup finishes before
``Engine.start_profile``; only the second batch enters the generated reports.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

# These values must be visible before SGLang loads either plugin entry point.
os.environ["SGLANG_FL_MODE"] = "platform_profile"
os.environ["SGLANG_PLATFORM"] = "sglang_fl"
os.environ["SGLANG_PLUGINS"] = "sglang_fl"
os.environ["USE_FLAGGEMS"] = "0"
os.environ["SGLANG_FL_PREFER"] = "vendor"
os.environ["SGLANG_FL_STRICT"] = "1"
os.environ.setdefault("SGLANG_PROFILE_WITH_STACK", "false")
os.environ.setdefault("SGLANG_PROFILE_RECORD_SHAPES", "true")

from tools.operator_profiling.environment import (  # noqa: E402
    validate_profiling_environment,
)
from tools.operator_profiling.inventory_report import (  # noqa: E402
    generate_inventory_report,
)
from tools.operator_profiling.trace_report import generate_reports  # noqa: E402
from tools.operator_profiling.workload import (  # noqa: E402
    build_exact_token_ids,
    generation_text_digest,
    generation_token_prefixes,
    resolve_lengths,
    resolve_runtime_reserved_tokens,
    validate_generation_result,
)


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return result


def _input_lengths(value: str) -> list[int]:
    try:
        lengths = [_positive_int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not lengths:
        raise argparse.ArgumentTypeError("at least one input length is required")
    return list(dict.fromkeys(lengths))


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    temporary.replace(path)


def _runtime_metadata(engine: Any, args: argparse.Namespace) -> dict[str, Any]:
    import torch

    server_info = engine.get_server_info()
    selected = {
        name: server_info.get(name)
        for name in (
            "model_path",
            "context_length",
            "max_req_input_len",
            "tp_size",
            "max_running_requests",
            "effective_max_running_requests_per_dp",
            "chunked_prefill_size",
            "attention_backend",
            "enable_torch_compile",
            "disable_cuda_graph",
            "disable_piecewise_cuda_graph",
            "disable_radix_cache",
            "random_seed",
        )
        if name in server_info
    }
    cli = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }
    return {
        "profile_mode": os.environ["SGLANG_FL_MODE"],
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "sglang_version": _distribution_version("sglang"),
        "sglang_fl_version": _distribution_version("sglang_fl"),
        "flagtree_version": _distribution_version("flagtree"),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "server": selected,
        "cli": cli,
        "adaptation": {
            "layer1_flaggems": False,
            "layer2_plugin_dispatch": True,
            "layer2_flagos_implementations": False,
            "layer2_vendor_preferred": True,
            "layer2_reference_fallback": False,
            "layer2_sglang_framework_fallback": True,
            "layer3_platform_runtime": True,
            "vendor_patches": True,
        },
    }


def _context_length(engine: Any) -> int:
    value = getattr(engine.tokenizer_manager, "context_len", None)
    if not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"SGLang returned an invalid context length: {value!r}")
    return value


def _reserved_tokens(engine: Any) -> int:
    tokenizer_manager = engine.tokenizer_manager
    framework_reserved = getattr(tokenizer_manager, "num_reserved_tokens", 0)
    if not isinstance(framework_reserved, int) or framework_reserved < 0:
        framework_reserved = 0
    max_req_input_len = getattr(tokenizer_manager, "max_req_input_len", None)
    if not isinstance(max_req_input_len, int):
        max_req_input_len = None
    return resolve_runtime_reserved_tokens(
        _context_length(engine),
        framework_reserved_tokens=framework_reserved,
        max_req_input_len=max_req_input_len,
    )


def _run_batch(
    engine: Any,
    prompt_ids: list[int],
    concurrency: int,
    output_tokens: int,
) -> tuple[Any, float]:
    prompts = [prompt_ids] * concurrency
    sampling_params = {
        "temperature": 0,
        "max_new_tokens": output_tokens,
        "ignore_eos": True,
    }
    started = time.perf_counter()
    result = engine.generate(input_ids=prompts, sampling_params=sampling_params)
    elapsed = time.perf_counter() - started
    validate_generation_result(
        result, concurrency=concurrency, output_tokens=output_tokens
    )
    return result, elapsed


def _profile_case(
    engine: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    run_dir: Path,
    requested_input_tokens: int,
    common_metadata: dict[str, Any],
) -> dict[str, Any]:
    resolution = resolve_lengths(
        requested_input_tokens,
        args.output_tokens,
        _context_length(engine),
        reserved_tokens=_reserved_tokens(engine),
        overflow_policy=args.length_overflow_policy,
    )
    prompt_ids = build_exact_token_ids(tokenizer, resolution.actual_input_tokens)
    if len(prompt_ids) != resolution.actual_input_tokens:
        raise RuntimeError(
            f"prompt builder produced {len(prompt_ids)} tokens, "
            f"expected {resolution.actual_input_tokens}"
        )

    case_name = (
        f"input_{requested_input_tokens}_actual_{resolution.actual_input_tokens}_"
        f"output_{args.output_tokens}_concurrency_{args.concurrency}"
    )
    case_dir = run_dir / case_name
    trace_dir = case_dir / "traces"
    audit_dir = case_dir / "audit" / "operator_report"
    result_dir = case_dir / "results"
    trace_dir.mkdir(parents=True)
    profile_id = f"sglang-fl-eager-{case_name}"

    print(
        f"[{case_name}] warmup starts: actual_input={resolution.actual_input_tokens} "
        f"output={resolution.actual_output_tokens} concurrency={args.concurrency}",
        flush=True,
    )
    warmup_result, warmup_seconds = _run_batch(
        engine,
        prompt_ids,
        args.concurrency,
        resolution.actual_output_tokens,
    )
    engine.flush_cache()

    profiler_started = False
    profiled_result = None
    profiled_seconds = 0.0
    try:
        engine.start_profile(
            output_dir=str(trace_dir),
            activities=["CPU", "GPU"],
            with_stack=False,
            record_shapes=True,
            profile_prefix=profile_id,
            merge_profiles=False,
        )
        profiler_started = True
        print(f"[{case_name}] profiled batch starts", flush=True)
        profiled_result, profiled_seconds = _run_batch(
            engine,
            prompt_ids,
            args.concurrency,
            resolution.actual_output_tokens,
        )
    finally:
        if profiler_started:
            engine.stop_profile()

    warmup_digest = generation_text_digest(warmup_result)
    profiled_digest = generation_text_digest(profiled_result)
    if warmup_digest != profiled_digest:
        raise RuntimeError(
            "profiled output differs from the deterministic warmup output: "
            f"warmup={warmup_digest} profiled={profiled_digest}"
        )

    workload = {
        **resolution.to_dict(),
        "concurrency": args.concurrency,
        "warmup_included": False,
        "warmup_seconds": warmup_seconds,
        "profiled_wall_time_seconds": profiled_seconds,
        "total_input_tokens": resolution.actual_input_tokens * args.concurrency,
        "total_output_tokens": resolution.actual_output_tokens * args.concurrency,
        "input_tokens_per_second": (
            resolution.actual_input_tokens * args.concurrency / profiled_seconds
        ),
        "output_tokens_per_second": (
            resolution.actual_output_tokens * args.concurrency / profiled_seconds
        ),
        "warmup_output_sha256": warmup_digest,
        "profiled_output_sha256": profiled_digest,
        "profiled_output_token_prefixes": generation_token_prefixes(
            profiled_result, prefix_tokens=10
        ),
        "profile_id": profile_id,
    }
    metadata = {**common_metadata, "workload": workload}
    _atomic_json(case_dir / "workload.json", metadata)
    audit_summary = generate_reports(
        trace_dir,
        audit_dir,
        profile_id=profile_id,
        metadata=metadata,
        jobs=min(args.tp_size, 4),
    )
    summary = generate_inventory_report(audit_dir, result_dir)

    expected_ranks = set(range(args.tp_size))
    actual_ranks = {int(rank) for rank in audit_summary["per_rank"]}
    if actual_ranks != expected_ranks:
        raise RuntimeError(
            f"expected profiler traces for TP ranks {sorted(expected_ranks)}, "
            f"found {sorted(actual_ranks)}"
        )
    if summary["compute_kernel_event_count"] <= 0:
        raise RuntimeError("profile contained no compute kernel events")
    if not all(audit_summary["validation"].values()):
        raise RuntimeError(
            f"profiler audit validation failed: {audit_summary['validation']}"
        )
    if not all(summary["validation"].values()):
        raise RuntimeError(f"report validation failed: {summary['validation']}")

    (case_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
    print(
        f"[{case_name}] complete: kernels={summary['compute_kernel_event_count']} "
        f"results={result_dir}",
        flush=True,
    )
    return {
        "case": case_name,
        "case_dir": str(case_dir),
        "resolution": resolution.to_dict(),
        "profiled_wall_time_seconds": profiled_seconds,
        "runtime_kernel_event_count": audit_summary["runtime_kernel_event_count"],
        "compute_kernel_event_count": summary["compute_kernel_event_count"],
        "unique_stable_kernel_names": summary["unique_stable_kernel_names"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect target-platform SGLang operator names, shapes, provenance, "
            "and kernel time with self-developed compute replacements disabled"
        )
    )
    parser.add_argument("--model-path", default="/models/Qwen3.6-27B")
    parser.add_argument("--tp-size", type=_positive_int, default=4)
    parser.add_argument(
        "--input-tokens",
        type=_input_lengths,
        default=[1024, 4096, 16384],
        help="comma-separated requested input lengths",
    )
    parser.add_argument("--output-tokens", type=_positive_int, default=1024)
    parser.add_argument("--concurrency", type=_positive_int, default=64)
    parser.add_argument(
        "--length-overflow-policy",
        choices=("truncate_input", "error"),
        default="truncate_input",
    )
    parser.add_argument("--mem-fraction-static", type=float, default=0.80)
    parser.add_argument("--chunked-prefill-size", type=_positive_int, default=8192)
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--require-validated-versions",
        action="store_true",
        help="reject versions outside the documented H20 validation baseline",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not 0 < args.mem_fraction_static < 1:
        raise ValueError("--mem-fraction-static must be between zero and one")

    preflight = validate_profiling_environment(
        model_path=model_path,
        tp_size=args.tp_size,
        repository_root=_REPOSITORY_ROOT,
        require_validated_versions=args.require_validated_versions,
    )
    print("OPERATOR_PROFILING_PREFLIGHT_PASS", flush=True)
    for warning in preflight["warnings"]:
        print(f"WARNING: {warning}", flush=True)

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True)

    from transformers import AutoTokenizer

    from sglang.srt.entrypoints.engine import Engine

    engine_kwargs: dict[str, Any] = {}
    if args.attention_backend:
        engine_kwargs["attention_backend"] = args.attention_backend
    engine = Engine(
        model_path=str(model_path),
        tp_size=args.tp_size,
        mem_fraction_static=args.mem_fraction_static,
        trust_remote_code=True,
        disable_cuda_graph=True,
        disable_piecewise_cuda_graph=True,
        enable_torch_compile=False,
        disable_radix_cache=True,
        allow_auto_truncate=False,
        max_running_requests=args.concurrency,
        chunked_prefill_size=args.chunked_prefill_size,
        random_seed=args.random_seed,
        **engine_kwargs,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        common_metadata = _runtime_metadata(engine, args)
        _atomic_json(run_dir / "run.json", common_metadata)
        cases = [
            _profile_case(
                engine,
                tokenizer,
                args,
                run_dir,
                input_tokens,
                common_metadata,
            )
            for input_tokens in args.input_tokens
        ]
        manifest = {**common_metadata, "run_id": run_id, "cases": cases}
        _atomic_json(run_dir / "manifest.json", manifest)
        (run_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
        print(f"EAGER_OPERATOR_PROFILE_PASS output={run_dir}")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"EAGER_OPERATOR_PROFILE_FAIL: {error}", file=sys.stderr)
        raise
