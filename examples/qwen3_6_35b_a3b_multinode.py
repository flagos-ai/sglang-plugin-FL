# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-35B-A3B (MoE) multi-node inference verification with sglang-plugin-FL.

Validates that sglang-plugin-FL correctly handles multi-node tensor parallelism
by launching a distributed SGLang server across 2 nodes and running text,
concurrent, multimodal (VL), and high-concurrency inference tests.

Supports CUDA, MUSA, and Ascend NPU; platform-specific server flags and env
vars are applied automatically at runtime.

============================================================================
Usage:
  This script runs as EITHER master (node_rank=0) or worker (node_rank=1),
  controlled by the --role argument.

  Step 1 — Start master on node 0 (e.g. 192.168.0.66):
    python examples/qwen3_6_35b_a3b_multinode.py --role master --master-addr 192.168.0.66 --tp 2 --pp 2

  Step 2 — Start worker on node 1 (e.g. 192.168.0.65):
    python examples/qwen3_6_35b_a3b_multinode.py --role worker --master-addr 192.168.0.66 --tp 2 --pp 2

  NOTE: Start master FIRST, then start worker within a few minutes.

Full tested command (2 nodes × 2 GPUs each, TP=2 PP=2):

  [Node 0 / Master / 192.168.0.66]
    CUDA_VISIBLE_DEVICES=0,1 \
    SGLANG_FL_DIST_BACKEND=flagcx \
    FLAGCX_PATH=/mine/FlagCX_v0.13.0 \
    SGLANG_FL_FLAGOS_BLACKLIST=count_nonzero \
    SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
    GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0 \
        python examples/qwen3_6_35b_a3b_multinode.py --role master --master-addr 192.168.0.66 --tp 2 --pp 2

  [Node 1 / Worker / 192.168.0.65]
    CUDA_VISIBLE_DEVICES=0,1 \
    SGLANG_FL_DIST_BACKEND=flagcx \
    FLAGCX_PATH=/mine/FlagCX_v0.13.0 \
    SGLANG_FL_FLAGOS_BLACKLIST=count_nonzero \
    SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
    GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0 \
        python examples/qwen3_6_35b_a3b_multinode.py --role worker --master-addr 192.168.0.66 --tp 2 --pp 2

  Total GPUs: TP × PP = 2 × 2 = 4 (2 per node)

  On MUSA, swap CUDA_VISIBLE_DEVICES → MUSA_VISIBLE_DEVICES; on Ascend NPU,
  use ASCEND_RT_VISIBLE_DEVICES. SGLANG_FL_DIST_BACKEND=flagcx is recommended
  on both non-CUDA platforms.

Environment variables:
  MODEL_PATH       Model path (default: /models/Qwen3.6-35B-A3B)
  CUDA_VISIBLE_DEVICES  GPU selection on CUDA (e.g. 0,1)
  MUSA_VISIBLE_DEVICES  Device selection on MUSA
  ASCEND_RT_VISIBLE_DEVICES  Device selection on Ascend NPU
  GLOO_SOCKET_IFNAME    Network interface for Gloo (default: eth0)
  NCCL_SOCKET_IFNAME    Network interface for NCCL (default: eth0)
  SGLANG_FL_DIST_BACKEND  Communication backend (flagcx / nccl)
  FLAGCX_PATH           Path to FlagCX installation
  SGLANG_FL_FLAGOS_BLACKLIST         Ops to exclude from FlagGems
  SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK  Set to 0 to skip memory check

Supported TP sizes: 1, 2, 4, 8, 16 (num_attention_heads=16)
============================================================================
"""

import argparse
import base64
import concurrent.futures
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
import torch

# ─── Platform detection ──────────────────────────────────────────────────────

_is_txda = hasattr(torch, "txda") and torch.txda.is_available()
_is_musa = hasattr(torch, "musa") and torch.musa.is_available()
_is_npu = hasattr(torch, "npu") and torch.npu.is_available()

if _is_txda:
    os.environ.setdefault("SGLANG_FL_TIMER_ENABLE", "1")
    os.environ.setdefault("SGLANG_REQ_WAITING_TIMEOUT", "-1")
    os.environ.setdefault("SGLANG_REQ_RUNNING_TIMEOUT", "-1")
if _is_npu:
    os.environ.setdefault("SGLANG_ENABLE_OVERLAP_PLAN_STREAM", "0")
    os.environ.setdefault("SGLANG_ENABLE_SPEC_V2", "1")
    os.environ.setdefault("HCCL_BUFFSIZE", "2400")
    os.environ.setdefault("SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "128")
elif _is_musa:
    os.environ.setdefault("MCCL_IB_DISABLE", "1")
    
# Extra launch_server flags per platform.
# - MUSA: page_size=1 works around a sglang platform bug.
# - Ascend NPU: requires ascend attention backend, bfloat16, radix cache off.
if _is_musa:
    _PLATFORM_SERVER_ARGS: list = ["--page-size", "1"]
elif _is_npu:
    _PLATFORM_SERVER_ARGS = [
        "--attention-backend", "ascend",
        "--device", "npu",
        "--dtype", "bfloat16",
        "--disable-radix-cache",
    ]
else:
    _PLATFORM_SERVER_ARGS = []

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/Qwen3.6-35B-A3B")

_HERE = Path(__file__).resolve().parent
IMG_DIR = Path(os.environ.get("IMAGE_DIR", _HERE / "test_images"))


# ─── Argument parsing ────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3.6-35B-A3B multi-node verification"
    )
    parser.add_argument(
        "--role",
        choices=["master", "worker"],
        required=True,
        help="Node role: 'master' (node_rank=0, runs tests) or 'worker' (node_rank=1)",
    )
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=2, help="Pipeline parallelism size")
    parser.add_argument(
        "--port", type=int, default=30000, help="API port (master only)"
    )
    parser.add_argument(
        "--master-addr", required=True, help="Master node IP (required)"
    )
    parser.add_argument(
        "--dist-port", type=int, default=20000, help="torch.distributed rendezvous port"
    )
    parser.add_argument("--nccl-port", type=int, default=28765, help="NCCL comm port")
    parser.add_argument("--nnodes", type=int, default=2, help="Number of nodes")
    parser.add_argument(
        "--node-rank",
        type=int,
        default=None,
        help="Override node rank (default: 0 for master, 1 for worker)",
    )
    parser.add_argument(
        "--max-wait",
        type=int,
        default=600,
        help="Max seconds to wait for server ready (master only)",
    )
    parser.add_argument(
        "--text-concurrency",
        type=int,
        default=32,
        help="Number of concurrent text requests for high-concurrency test",
    )
    parser.add_argument(
        "--vl-concurrency",
        type=int,
        default=8,
        help="Number of concurrent VL requests for high-concurrency test",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=300,
        help="HTTP request and result timeout in seconds (default: 300)",
    )
    return parser.parse_args()


# ─── HTTP helpers ────────────────────────────────────────────────────────────


def wait_for_server(port: int, timeout: int) -> bool:
    """Poll server health endpoint until ready or timeout."""
    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
        elapsed = int(time.time() - start)
        print(f"\r  {elapsed}s elapsed...", end="", flush=True)
    return False


def chat_request(
    port: int, prompt: str, max_tokens: int = 32, timeout: int = 300
) -> str:
    """Send a text chat completion request."""
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"


def vl_request(
    port: int, img_path: Path, question: str, max_tokens: int = 16, timeout: int = 300
) -> str:
    """Send a vision-language chat request with a base64-encoded image."""
    url = f"http://localhost:{port}/v1/chat/completions"
    mime = "image/png" if img_path.suffix == ".png" else "image/jpeg"
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    payload = {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {"type": "text", "text": question},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"


# ─── Test runner ─────────────────────────────────────────────────────────────


class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.total = 0

    def check(self, desc: str, expected: str, content: str):
        self.total += 1
        if expected.lower() in content.lower():
            print(f"  PASS: {desc}")
            self.passed += 1
        else:
            print(f"  FAIL: {desc}")
            print(f"        expected '{expected}' in: {content}")
            self.failed += 1

    def summary(self, tp_size: int, nnodes: int):
        print()
        print("=" * 56)
        print(f"  Results: {self.passed}/{self.total} passed, {self.failed} failed")
        print(f"  Model:   Qwen3.6-35B-A3B  TP: {tp_size}  Nodes: {nnodes}")
        print("=" * 56)
        if self.failed == 0:
            print("ALL TESTS PASSED")
        else:
            print("SOME TESTS FAILED")
        return self.failed == 0


def run_tests(
    port: int,
    tp_size: int,
    nnodes: int,
    text_concurrency: int = 32,
    vl_concurrency: int = 8,
    request_timeout: int = 300,
) -> bool:
    """Run all verification tests. Returns True if all passed."""
    t = TestRunner()

    # Test 1: Text Inference (Sequential)
    print("\n=== Test 1: Text Inference (Sequential) ===")
    r = chat_request(
        port,
        "How many states are there in the United States? Answer with just the number.",
        10,
        timeout=request_timeout,
    )
    print(f"  Q: How many states?  A: {r}")
    t.check("US states = 50", "50", r)

    r = chat_request(port, "What is the capital of France? Answer with one word.", 10, timeout=request_timeout)
    print(f"  Q: Capital of France?  A: {r}")
    t.check("Capital of France = Paris", "Paris", r)

    r = chat_request(port, "What is 2+3? Answer with just the number.", 10, timeout=request_timeout)
    print(f"  Q: 2+3?  A: {r}")
    t.check("2+3 = 5", "5", r)

    # Test 2: Longer Generation
    print("\n=== Test 2: Longer Generation ===")
    r = chat_request(port, "List the first 5 prime numbers, separated by commas.", 64, timeout=request_timeout)
    print(f"  Q: First 5 primes  A: {r}")
    t.check("Contains '2'", "2", r)
    t.check("Contains '7'", "7", r)

    # Test 3: Concurrent Text x4
    print("\n=== Test 3: Concurrent Text x4 ===")
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for i in range(1, 5):
            f = executor.submit(
                chat_request, port, f"What is {i}+{i}? Answer with just the number.", 10, request_timeout
            )
            futures[i] = f

        all_ok = True
        for i in range(1, 5):
            r = futures[i].result(timeout=request_timeout)
            expected = str(i + i)
            ok = expected in r
            print(f"  {'PASS' if ok else 'FAIL'}: {i}+{i}={expected} -> {r}")
            if not ok:
                all_ok = False

    t.total += 1
    if all_ok:
        t.passed += 1
    else:
        t.failed += 1
    print(f"  Time: {time.time() - t0:.1f}s")

    # Test 4: High-Concurrency Text
    print(f"\n=== Test 4: High-Concurrency Text x{text_concurrency} ===")
    t0 = time.time()
    ok_count = 0
    fail_count = 0
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=text_concurrency
    ) as executor:
        futures = {}
        for i in range(text_concurrency):
            a, b = i + 1, i + 2
            f = executor.submit(
                chat_request, port, f"What is {a}+{b}? Answer with just the number.", 10, request_timeout
            )
            futures[i] = (a, b, f)

        for i, (a, b, f) in futures.items():
            try:
                r = f.result(timeout=request_timeout)
                expected = str(a + b)
                if expected in r:
                    ok_count += 1
                else:
                    fail_count += 1
                    print(f"  FAIL: {a}+{b}={expected} -> {r}")
            except Exception as e:
                fail_count += 1
                print(f"  FAIL: {a}+{b} -> ERROR: {e}")

    t.total += 1
    if fail_count == 0:
        print(f"  PASS: {ok_count}/{text_concurrency} correct")
        t.passed += 1
    else:
        print(f"  FAIL: {ok_count}/{text_concurrency} correct, {fail_count} failed")
        t.failed += 1
    print(f"  Time: {time.time() - t0:.1f}s")

    # Test 5: Multimodal (VL) Sequential
    print("\n=== Test 5: Multimodal (VL) Sequential ===")
    if IMG_DIR.is_dir():
        r = vl_request(
            port,
            IMG_DIR / "red_square.jpg",
            "What color is shown in this image? Answer with one word.",
            10,
            timeout=request_timeout,
        )
        print(f"  Q: Color of square?  A: {r}")
        t.check("VL: red square = red", "red", r)

        r = vl_request(
            port,
            IMG_DIR / "cat.jpg",
            "What animal is in this image? Answer with one word.",
            10,
            timeout=request_timeout,
        )
        print(f"  Q: Animal in image?  A: {r}")
        t.check("VL: cat image = cat", "cat", r)

        r = vl_request(
            port,
            IMG_DIR / "digit_seven.png",
            "What digit is shown in this image? Answer with one digit.",
            10,
            timeout=request_timeout,
        )
        print(f"  Q: Digit in image?  A: {r}")
        t.check("VL: digit = 7", "7", r)
    else:
        print(f"  SKIP: test_images directory not found at {IMG_DIR}")

    # Test 6: High-Concurrency VL
    if IMG_DIR.is_dir() and vl_concurrency > 1:
        print(f"\n=== Test 6: High-Concurrency VL x{vl_concurrency} ===")
        vl_cases = [
            (IMG_DIR / "red_square.jpg", "What color is this? One word.", "red"),
            (IMG_DIR / "cat.jpg", "What animal? One word.", "cat"),
            (IMG_DIR / "digit_seven.png", "What digit? Just the number.", "7"),
            (IMG_DIR / "stop_sign.png", "What does this sign say? One word.", "stop"),
        ]
        # Cycle through images
        all_cases = (vl_cases * ((vl_concurrency // len(vl_cases)) + 1))[
            :vl_concurrency
        ]
        t0 = time.time()
        ok_count = 0
        fail_count = 0
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=vl_concurrency
        ) as executor:
            futures = {}
            for i, (img, q, exp) in enumerate(all_cases):
                futures[i] = (
                    img.name,
                    exp,
                    executor.submit(vl_request, port, img, q, 10, request_timeout),
                )
            for i, (name, expected, f) in futures.items():
                try:
                    r = f.result(timeout=request_timeout)
                    if expected.lower() in r.lower():
                        ok_count += 1
                    else:
                        fail_count += 1
                        print(f"  FAIL: {name} -> {r}")
                except Exception as e:
                    fail_count += 1
                    print(f"  FAIL: {name} -> ERROR: {e}")

        t.total += 1
        if fail_count == 0:
            print(f"  PASS: {ok_count}/{vl_concurrency} correct")
            t.passed += 1
        else:
            print(f"  FAIL: {ok_count}/{vl_concurrency} correct, {fail_count} failed")
            t.failed += 1
        print(f"  Time: {time.time() - t0:.1f}s")

    return t.summary(tp_size, nnodes)


# ─── Master logic ────────────────────────────────────────────────────────────


def run_master(args):
    """Launch server, wait for ready, run tests, shutdown."""
    node_rank = args.node_rank if args.node_rank is not None else 0

    print("=" * 56)
    print("  sglang-plugin-FL Multi-Node Verification")
    print(f"  Role:   MASTER (node_rank={node_rank})")
    print(f"  Model:  {MODEL_PATH}")
    print(f"  TP:     {args.tp}    PP: {args.pp}    Nodes: {args.nnodes}")
    print(f"  Master: {args.master_addr}  dist={args.dist_port}  nccl={args.nccl_port}")
    print(f"  API:    http://localhost:{args.port}")
    print(
        f"  Text concurrency: {args.text_concurrency}  VL concurrency: {args.vl_concurrency}  Request timeout: {args.request_timeout}s"
    )
    print("=" * 56)

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_PATH,
        "--tp",
        str(args.tp),
        "--pp-size",
        str(args.pp),
        "--port",
        str(args.port),
        "--nnodes",
        str(args.nnodes),
        "--node-rank",
        str(node_rank),
        "--dist-init-addr",
        f"{args.master_addr}:{args.dist_port}",
        "--nccl-port",
        str(args.nccl_port),
        "--mem-fraction-static",
        "0.85",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--trust-remote-code",
        *_PLATFORM_SERVER_ARGS,
    ]
    if _is_txda:
        insert_pos = cmd.index("--mem-fraction-static")
        cmd[insert_pos + 1] = "0.6"
        for flag in reversed([
            "--device", "txda",
            "--dtype", "bfloat16",
            "--disable-radix-cache",
            "--watchdog-timeout", "3600",
            "--mm-attention-backend", "triton_attn",
            "--disable-fast-image-processor",
            "--context-length", "8192",
            "--chunked-prefill-size", "256",
        ]):
            cmd.insert(insert_pos, flag)

    print("Launching server...")
    server_proc = subprocess.Popen(cmd)
    print(f"Server PID: {server_proc.pid}")

    def cleanup(signum=None, frame=None):
        print(f"\nShutting down server (PID={server_proc.pid})...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        print("Waiting for server...")
        if not wait_for_server(args.port, args.max_wait):
            print(f"\nTIMEOUT: Server not ready after {args.max_wait}s")
            cleanup()
            sys.exit(1)
        print("\nServer ready!")

        success = run_tests(
            args.port,
            args.tp,
            args.nnodes,
            text_concurrency=args.text_concurrency,
            vl_concurrency=args.vl_concurrency,
            request_timeout=args.request_timeout,
        )
        cleanup()
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\nERROR: {e}")
        cleanup()
        sys.exit(1)


# ─── Worker logic ────────────────────────────────────────────────────────────


def run_worker(args):
    """Launch worker that joins master's distributed group."""
    node_rank = args.node_rank if args.node_rank is not None else 1

    print("=" * 56)
    print("  sglang-plugin-FL Multi-Node Verification")
    print(f"  Role:   WORKER (node_rank={node_rank})")
    print(f"  Model:  {MODEL_PATH}")
    print(f"  TP:     {args.tp}    PP: {args.pp}    Nodes: {args.nnodes}")
    print(f"  Master: {args.master_addr}  dist={args.dist_port}  nccl={args.nccl_port}")
    print("=" * 56)

    # Connectivity check
    print(f"Checking connectivity to master ({args.master_addr}:{args.dist_port})...")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect((args.master_addr, args.dist_port))
        s.close()
        print("  Master reachable!")
    except Exception:
        print("  WARNING: Cannot connect to master (may not be ready yet, will retry)")

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_PATH,
        "--tp",
        str(args.tp),
        "--pp-size",
        str(args.pp),
        "--nnodes",
        str(args.nnodes),
        "--node-rank",
        str(node_rank),
        "--dist-init-addr",
        f"{args.master_addr}:{args.dist_port}",
        "--nccl-port",
        str(args.nccl_port),
        "--mem-fraction-static",
        "0.85",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--trust-remote-code",
        *_PLATFORM_SERVER_ARGS,
    ]
    if _is_txda:
        insert_pos = cmd.index("--mem-fraction-static")
        cmd[insert_pos + 1] = "0.6"
        for flag in reversed([
            "--device", "txda",
            "--dtype", "bfloat16",
            "--disable-radix-cache",
            "--watchdog-timeout", "3600",
            "--mm-attention-backend", "triton_attn",
            "--disable-fast-image-processor",
            "--context-length", "8192",
            "--chunked-prefill-size", "256",
        ]):
            cmd.insert(insert_pos, flag)

    print("Starting worker node... (will block until master shuts down)\n")
    try:
        result = subprocess.run(cmd)
        print("Worker node exited.")
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nWorker interrupted.")
        sys.exit(0)


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if not os.path.isdir(MODEL_PATH):
        print(f"ERROR: Model not found: {MODEL_PATH}")
        print("Set MODEL_PATH environment variable to the correct path.")
        sys.exit(1)

    # Ensure network interfaces are set
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "eth0")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")

    if args.role == "master":
        run_master(args)
    else:
        run_worker(args)