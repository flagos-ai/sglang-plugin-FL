# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Record and compare Qwen3.6 C1/C4 repeats in separately launched runtimes.

This is an opt-in diagnostic, not a performance or formal accuracy benchmark.
The caller selects the reference/candidate image and plugin environment.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import math
import os
import subprocess
import uuid
from pathlib import Path

PROMPTS = [
    "The capital of France is",
    "What is 17 multiplied by 13?",
    "What is the largest planet in the solar system?",
    "How many states are there in the United States?",
]


def read_engine_config(path, model_path=None, eager=False):
    import yaml

    config = dict(yaml.safe_load(Path(path).read_text())["llm"])
    config["model_path"] = model_path or config.pop("model")
    config.pop("model", None)
    if eager:
        config["disable_cuda_graph"] = True
    return config


def normalize_output(output):
    """Read token IDs from native SGLang output logprobs, never retokenize text."""
    meta = output["meta_info"]
    pairs = meta.get("output_token_logprobs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("missing output_token_logprobs; token parity is unverified")
    token_ids, logprobs = [], []
    for entry in pairs:
        probability, token_id = entry[:2]
        if type(token_id) is not int or not isinstance(probability, (float, int)):
            raise ValueError("invalid token/logprob entry")
        if not math.isfinite(probability):
            raise ValueError("non-finite output logprob")
        token_ids.append(token_id)
        logprobs.append(probability)
    if len(token_ids) != meta.get("completion_tokens"):
        raise ValueError("incomplete output token/logprob record")
    return {
        "text": output["text"],
        "token_ids": token_ids,
        "logprobs": logprobs,
        "server_request_id": meta.get("id"),
        "finish_reason": meta.get("finish_reason"),
    }


async def run_wave(engine, prompts, sampling, round_id, concurrency, start, run_id):
    indices = range(start, min(start + concurrency, len(prompts)))
    requests = [
        (index, f"{run_id}-r{round_id}-c{concurrency}-p{index}") for index in indices
    ]
    outputs = await asyncio.gather(
        *(
            engine.async_generate(
                prompt=prompts[index],
                sampling_params=sampling,
                return_logprob=True,
                logprob_start_len=-1,
                rid=request_id,
            )
            for index, request_id in requests
        ),
        return_exceptions=True,
    )
    records = []
    for (index, request_id), output in zip(requests, outputs):
        record = {
            "round": round_id,
            "concurrency": concurrency,
            "prompt_index": index,
            "request_id": request_id,
        }
        try:
            if isinstance(output, BaseException):
                raise output
            record.update(normalize_output(output))
            if record["server_request_id"] != request_id:
                raise ValueError(
                    "response request ID does not match the submitted request"
                )
            if len(record["token_ids"]) != sampling["max_new_tokens"]:
                raise ValueError(
                    "response did not complete the requested fixed token count"
                )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
    return records


def runtime_metadata():
    packages = {}
    for name in (
        "sglang",
        "sglang_fl",
        "torch",
        "torch_musa",
        "triton",
        "flag_gems",
        "mate",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "validation_checkout_commit": result.stdout.strip()
        if result.returncode == 0
        else None,
        "packages": packages,
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("SGLANG_MUSA_")
            or key
            in (
                "SGLANG_PLUGINS",
                "USE_FLAGGEMS",
                "SGLANG_FL_FLAGOS_BLACKLIST",
            )
        },
    }


def write_record(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def record_run(args):
    from sglang.srt.entrypoints.engine import Engine

    output = Path(args.output)
    if output.exists():
        raise ValueError(f"result already exists: {output}; choose a new output path")
    config = read_engine_config(args.engine_config, args.model_path, args.eager)
    prompts = json.loads(Path(args.prompts).read_text()) if args.prompts else PROMPTS
    if (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(prompt, str) and prompt for prompt in prompts)
    ):
        raise ValueError("prompts must be a non-empty JSON list of non-empty strings")
    if len(prompts) < max(args.concurrency):
        raise ValueError("provide at least as many prompts as the largest concurrency")
    sampling = {
        "temperature": 0,
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": True,
    }
    data = {
        "schema_version": 1,
        "label": args.label,
        "status": "running",
        "identity": {
            "image_digest": args.image_digest,
            "mate_revision": args.mate_revision,
            "plugin_commit": args.plugin_commit,
        },
        "runtime": runtime_metadata(),
        "protocol": {
            "engine": config,
            "prompts": prompts,
            "sampling": sampling,
            "rounds": args.rounds,
            "concurrency": args.concurrency,
            "warmup": 1,
        },
        "records": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_record(output, data)
    engine = None
    try:
        engine = Engine(**config)
        if args.label == "candidate":
            import sglang_fl

            if not (sglang_fl.is_plugin_loaded() and sglang_fl.is_plugin_active()):
                raise RuntimeError("candidate plugin was not loaded and activated")
            data["runtime"]["installed_plugin_path"] = sglang_fl.__file__
        # Warmup is intentionally excluded from comparisons.
        engine.generate(prompt=prompts[0], sampling_params=sampling)
        run_id = uuid.uuid4().hex
        for round_id in range(args.rounds):
            for concurrency in args.concurrency:
                for start in range(0, len(prompts), concurrency):
                    records = engine.loop.run_until_complete(
                        run_wave(
                            engine,
                            prompts,
                            sampling,
                            round_id,
                            concurrency,
                            start,
                            run_id,
                        )
                    )
                    data["records"].extend(records)
                    write_record(output, data)
                    if any("error" in row for row in records):
                        raise RuntimeError(
                            "request failed; partial records have been preserved"
                        )
        data["status"] = "complete"
    except BaseException as exc:
        data["status"] = "failed"
        data["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_record(output, data)
        if engine is not None:
            engine.shutdown()
    print(f"Recorded {len(data['records'])} requests in {output}")


def index_records(data):
    if data.get("schema_version") != 1 or data.get("status") != "complete":
        raise ValueError("record is incomplete or has an unsupported schema")
    protocol = data["protocol"]
    if protocol["rounds"] < 2:
        raise ValueError("at least two rounds are required to verify repeatability")
    expected = {
        (round_id, concurrency, index)
        for round_id in range(protocol["rounds"])
        for concurrency in protocol["concurrency"]
        for index in range(len(protocol["prompts"]))
    }
    rows = {}
    for row in data["records"]:
        key = row["round"], row["concurrency"], row["prompt_index"]
        if (
            "error" in row
            or key in rows
            or not row.get("token_ids")
            or (len(row["token_ids"]) != len(row.get("logprobs", [])))
            or not all(type(value) is int for value in row["token_ids"])
            or not all(
                isinstance(value, (float, int)) and math.isfinite(value)
                for value in row["logprobs"]
            )
        ):
            raise ValueError(f"invalid or duplicate request record: {key}")
        rows[key] = row
    if not expected or set(rows) != expected:
        raise ValueError("request records do not cover the declared protocol")
    return rows


def compare_records(reference, candidate):
    if reference.get("label") != "reference" or candidate.get("label") != "candidate":
        raise ValueError(
            "compare requires a reference record followed by a candidate record"
        )
    if reference["protocol"] != candidate["protocol"]:
        raise ValueError(
            "protocols differ (including engine settings); not a controlled comparison"
        )
    reference_rows, candidate_rows = index_records(reference), index_records(candidate)

    def differences(pairs):
        mismatches = []
        for key, left, right in pairs:
            fields = [
                name for name in ("token_ids", "logprobs") if left[name] != right[name]
            ]
            if fields:
                mismatches.append({"request": key, "fields": fields})
        return mismatches

    result = {}
    for label, rows in (
        ("reference_self_drift", reference_rows),
        ("candidate_self_drift", candidate_rows),
    ):
        result[label] = differences(
            (key, rows[(0, key[1], key[2])], row)
            for key, row in rows.items()
            if key[0] > 0
        )
    result["between_services"] = differences(
        (key, row, candidate_rows[key]) for key, row in reference_rows.items()
    )
    return result


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record")
    record.add_argument("--label", choices=("reference", "candidate"), required=True)
    record.add_argument("--engine-config", required=True)
    record.add_argument("--model-path", default=os.getenv("MODEL_PATH"))
    record.add_argument("--eager", action="store_true")
    record.add_argument(
        "--prompts", help="JSON list of rendered prompt strings shared by both arms"
    )
    record.add_argument("--rounds", type=positive_int, default=3)
    record.add_argument("--concurrency", type=positive_int, nargs="+", default=[1, 4])
    record.add_argument("--max-new-tokens", type=positive_int, default=32)
    record.add_argument("--image-digest", required=True)
    record.add_argument("--mate-revision", required=True)
    record.add_argument(
        "--plugin-commit",
        required=True,
        help="installed plugin commit, not the validation checkout",
    )
    record.add_argument("--output", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("reference")
    compare.add_argument("candidate")
    args = parser.parse_args()
    try:
        if args.command == "record":
            if not all(
                value.strip()
                for value in (
                    args.image_digest,
                    args.mate_revision,
                    args.plugin_commit,
                )
            ):
                raise ValueError(
                    "image, MATE and installed plugin identities must not be empty"
                )
            if args.rounds < 2 or len(args.concurrency) != len(set(args.concurrency)):
                raise ValueError(
                    "use at least two rounds and distinct concurrency values"
                )
            record_run(args)
            return 0
        result = compare_records(
            json.loads(Path(args.reference).read_text()),
            json.loads(Path(args.candidate).read_text()),
        )
        print(json.dumps(result, indent=2))
        return int(any(result.values()))
    except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Validation error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
