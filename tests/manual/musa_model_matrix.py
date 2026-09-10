# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Run real model serving checks on MUSA, independently of examples.

Run inside a prepared MUSA environment, in tmux on remote machines. Servers
run sequentially and only their own process groups are terminated afterwards.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


CASES = {
    "phi4": ("Phi-4-mini-instruct", 1, False),
    "gemma3": ("rnj-1-instruct", 1, False),
    "cohere": ("aya-23-8B", 1, False),
    "qwen36_dense": ("Qwen3.6-27B", 4, True),
    "qwen36_moe": ("Qwen3.6-35B-A3B", 4, True),
}


def request(url, payload=None, timeout=300):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read()
        return json.loads(body) if body else {}


def chat(url, content, expected):
    result = request(
        url + "/v1/chat/completions",
        {
            "model": "matrix-model",
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    answer = result["choices"][0]["message"]["content"]
    assert expected.lower() in answer.lower(), result
    assert result["usage"]["completion_tokens"] > 0, result
    return {"answer": answer, "usage": result["usage"]}


def decode(url, sampling=False):
    params = {
        "temperature": 0.7 if sampling else 0,
        "max_new_tokens": 64,
        "ignore_eos": True,
    }
    if sampling:
        params["top_p"] = 0.9
    result = request(
        url + "/generate",
        {
            "text": "The capital of France is",
            "sampling_params": params,
        },
    )
    assert result["meta_info"]["completion_tokens"] == 64, result
    assert result["text"].strip(), result
    if not sampling:
        assert "paris" in result["text"].lower(), result
    return {
        "text": result["text"],
        "completion_tokens": result["meta_info"]["completion_tokens"],
    }


def stop_server(process):
    # The server owns a new session, so this cannot stop unrelated jobs.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def run_case(name, args):
    model_name, tp, multimodal = CASES[name]
    if tp > 1:
        tp = args.large_model_tp
    model = args.model_root / model_name
    result = {
        "case": name,
        "model": str(model),
        "tp": tp,
        "status": "failed",
        "checks": {},
    }
    log_path = args.output_dir / f"{name}.log"
    result["log"] = str(log_path)
    process = None
    started = time.monotonic()
    try:
        with socket.socket() as probe:
            # A previous test leaves accepted connections in TIME_WAIT.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", args.port))
        config = json.loads((model / "config.json").read_text())
        result["architectures"] = config.get("architectures")
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(model),
            "--served-model-name",
            "matrix-model",
            "--tp-size",
            str(tp),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--trust-remote-code",
            "--attention-backend",
            "fa3",
            "--sampling-backend",
            "pytorch",
            "--page-size",
            "1",
            "--disable-radix-cache",
            "--context-length",
            "4096",
            "--chunked-prefill-size",
            "2048",
            "--max-running-requests",
            "8",
            "--mem-fraction-static",
            "0.65",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--cuda-graph-backend-decode",
            "full",
            "--cuda-graph-max-bs",
            "4",
        ]
        result["command"] = command
        url = f"http://127.0.0.1:{args.port}"
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited with code {process.returncode}")
                try:
                    request(url + "/health", timeout=3)
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(2)
            else:
                raise TimeoutError("Server startup timed out")
            result["startup_seconds"] = round(time.monotonic() - started, 2)
            checks = result["checks"]
            checks["capital"] = chat(
                url, "What is the capital of France? Answer with one word.", "Paris"
            )
            checks["arithmetic"] = chat(
                url, "What is 17 multiplied by 13? Answer with just the number.", "221"
            )
            if multimodal:
                image = (
                    Path(__file__).resolve().parents[2]
                    / "examples/test_images/red_square.jpg"
                )
                encoded = base64.b64encode(image.read_bytes()).decode()
                checks["image"] = chat(
                    url,
                    [
                        {
                            "type": "text",
                            "text": "What color is this image? Answer with one word.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + encoded},
                        },
                    ],
                    "red",
                )
            checks["decode_64"] = decode(url)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                checks["concurrent_decode_64"] = list(
                    pool.map(lambda _: decode(url), range(4))
                )
                checks["concurrent_sampling_64"] = list(
                    pool.map(lambda _: decode(url, True), range(4))
                )
            text = log_path.read_text(errors="replace")
            assert "Capture target decode CUDA graph end" in text, (
                "Decode graph was not captured"
            )
            assert "musa graph: True" in text, "No decode graph replay observed"
            checks["graph_replay"] = True
            result["status"] = "passed"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None:
            stop_server(process)
    result["elapsed_seconds"] = round(time.monotonic() - started, 2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--port", type=int, default=31818)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--large-model-tp", type=int, choices=(1, 2, 4, 8), default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name in ("sglang", "sglang_fl", "flagtree", "flag_gems", "torch", "torch_musa"):
        versions[name] = importlib.metadata.version(name)
    results = {
        "versions": versions,
        "module_origins": {
            name: importlib.util.find_spec(name).origin
            for name in ("sglang", "sglang_fl", "flag_gems", "triton")
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "MUSA_VISIBLE_DEVICES",
                "SGLANG_FL_PER_OP",
                "SGLANG_MUSA_FP32_TP_ALLREDUCE",
                "TORCH_COMPILE_DISABLE",
                "SGLANG_FL_FLAGOS_BLACKLIST",
            )
        },
        "results": [],
    }
    for name in args.models:
        result = run_case(name, args)
        results["results"].append(result)
        (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in ("case", "status", "elapsed_seconds", "checks", "error")
                    if k in result
                }
            ),
            flush=True,
        )
    return 0 if all(r["status"] == "passed" for r in results["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
