# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-35B-A3B (MoE) concurrent inference test with sglang-plugin-FL.

Tests text-only, multimodal, and mixed concurrent requests to verify that
the scheduling pipeline handles parallel workloads correctly.

Usage:
  python qwen3_6_35b_a3b_concurrent.py [--mode text|vl|mixed|all]

Modes:
  text    N-way concurrent text requests.
  vl      N-way concurrent multimodal requests.
  mixed   Text + VL requests mixed together concurrently.
  all     Run all modes in order (default).

Environment variables:
  MODEL_PATH       Model path (default: /models/Qwen3.6-35B-A3B)
  TP_SIZE          Tensor parallelism (default: 1)
  MAX_TOKENS       Max generation tokens for text (default: 256)
  CONCURRENT_N     Concurrent request count (default: 16)
  IMAGE_DIR        Test image directory (default: examples/test_images/)
"""

import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path
import torch

# ─── Platform detection ───────────────────────────────────────────────────────

_is_musa = hasattr(torch, "musa") and torch.musa.is_available()
_is_npu = hasattr(torch, "npu") and torch.npu.is_available()
_is_txda = hasattr(torch, "txda") and torch.txda.is_available()

# Must be set before importing sglang.
if _is_npu:
    os.environ.setdefault("SGLANG_ENABLE_OVERLAP_PLAN_STREAM", "0")
    os.environ.setdefault("SGLANG_ENABLE_SPEC_V2", "1")
    os.environ.setdefault("HCCL_BUFFSIZE", "2400")
    os.environ.setdefault("SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "128")

if _is_txda:
    os.environ.setdefault("SGLANG_FL_TIMER_ENABLE", "1")
    os.environ.setdefault("SGLANG_REQ_WAITING_TIMEOUT", "-1")
    os.environ.setdefault("SGLANG_REQ_RUNNING_TIMEOUT", "-1")

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/Qwen3.6-35B-A3B")
TP_SIZE = int(os.environ.get("TP_SIZE", "4" if _is_npu or _is_txda else "1"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))
CONCURRENT_N = int(os.environ.get("CONCURRENT_N", "16"))

_HERE = Path(__file__).resolve().parent
IMAGE_DIR = Path(os.environ.get("IMAGE_DIR", _HERE / "test_images"))

if _is_musa:
    _extra_engine_kwargs: dict = {"page_size": 1, "trust_remote_code": True}
elif _is_npu:
    _extra_engine_kwargs = {
        "attention_backend": "ascend",
        "device": "npu",
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "disable_radix_cache": True,
    }
elif _is_txda:
    # ─── Early stub-module injection ─────────────────────────────────────────
    try:
        from sglang_fl.dispatch.backends.vendor.tsingmicro.patches.platform_stubs import patch as _patch_stubs
        _patch_stubs()
    except Exception:
        pass
    _extra_engine_kwargs = {
        "device": "txda",
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "disable_radix_cache": True,
        "watchdog_timeout": 3600,
        "mm_attention_backend": "triton_attn",
        "disable_fast_image_processor": True,
        "context_length": 8192,
        "chunked_prefill_size":256
    }
else:
    _extra_engine_kwargs = {"trust_remote_code": True}

# ─── Test data ────────────────────────────────────────────────────────────────

TEXT_PROMPTS = [
    "How many states are there in the United States?",
    "The capital of France is",
    "What is the largest planet in the solar system?",
    "Who wrote the play Romeo and Juliet?",
    "What is 17 multiplied by 13?",
    "Name the three primary colors of light.",
    "What year did World War II end?",
    "Explain the concept of gravity in one sentence.",
]

TEXT_EXPECTED = {
    "How many states are there in the United States?": ["50"],
    "The capital of France is": ["paris"],
    "What is the largest planet in the solar system?": ["jupiter"],
    "Who wrote the play Romeo and Juliet?": ["shakespeare"],
    "What is 17 multiplied by 13?": ["221"],
    "Name the three primary colors of light.": ["red", "green", "blue"],
    "What year did World War II end?": ["1945"],
    "Explain the concept of gravity in one sentence.": ["mass", "force", "attract"],
}

VL_CASES = [
    {
        "image": "red_square.jpg",
        "question": "What color is shown in this image? Answer with one word.",
        "expected": ["red"],
    },
    {
        "image": "cat.jpg",
        "question": "What animal is in this image? Answer with one word.",
        "expected": ["cat"],
    },
    {
        "image": "stop_sign.png",
        "question": "What is the color of this sign? Answer with one word.",
        "expected": ["red"],
    },
    {
        "image": "digit_seven.png",
        "question": "What digit is shown in this image? Answer with one digit.",
        "expected": ["7"],
    },
]

ALL_MODES = ["text", "vl", "mixed"]

# ─── Prompt formatting ───────────────────────────────────────────────────────

_tokenizer = None
_processor = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return _tokenizer


def _get_processor():
    global _processor
    if _processor is None:
        from transformers import AutoProcessor

        _processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return _processor


def _text_prompt(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    return _get_tokenizer().apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def _vl_prompt(question: str, image_uri: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": question},
            ],
        }
    ]
    return _get_processor().apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def _image_uri(name: str) -> str:
    return f"file://{(IMAGE_DIR / name).resolve()}"


# ─── Engine factory ──────────────────────────────────────────────────────────


def _make_engine():
    from sglang.srt.entrypoints.engine import Engine

    return Engine(
        model_path=MODEL_PATH,
        tp_size=TP_SIZE,
        mem_fraction_static=0.6 if _is_txda else 0.85,
        disable_cuda_graph=True,
        disable_piecewise_cuda_graph=True,
        **_extra_engine_kwargs,
    )


# ─── Sampling params ─────────────────────────────────────────────────────────

_TEXT_SAMPLING = {"max_new_tokens": MAX_TOKENS, "temperature": 0}
_VL_SAMPLING = {"max_new_tokens": 64, "temperature": 0}

# ─── Stats helpers ───────────────────────────────────────────────────────────


def _percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _report(label, n_req, elapsed, latencies, total_tokens):
    mean = statistics.fmean(latencies) if latencies else 0.0
    p50 = _percentile(latencies, 0.50)
    p99 = _percentile(latencies, 0.99)
    print(
        f"\n[{label}] {n_req} req | wall {elapsed:.2f}s | "
        f"{n_req / elapsed:.2f} req/s | {total_tokens / elapsed:.1f} tok/s | "
        f"latency mean {mean:.2f}s P50 {p50:.2f}s P99 {p99:.2f}s"
    )


# ─── Concurrent runners ─────────────────────────────────────────────────────


def run_text_concurrent(engine):
    """N concurrent text requests."""
    base = [(p, _text_prompt(p)) for p in TEXT_PROMPTS]
    items = [base[i % len(base)] for i in range(CONCURRENT_N)]

    async def one(prompt):
        t0 = time.perf_counter()
        r = await engine.async_generate(prompt=prompt, sampling_params=_TEXT_SAMPLING)
        return time.perf_counter() - t0, r

    async def run():
        t0 = time.perf_counter()
        results = await asyncio.gather(*(one(p) for _, p in items))
        return time.perf_counter() - t0, results

    elapsed, results = engine.loop.run_until_complete(run())
    latencies = [lat for lat, _ in results]
    total_tokens = sum(
        r.get("meta_info", {}).get("completion_tokens", 0) for _, r in results
    )

    seen = set()
    for (label, _), (_, r) in zip(items, results):
        if label in seen:
            continue
        seen.add(label)
        print(f"  {label!r}\n    → {r['text']!r}")

    _report("text-concurrent", CONCURRENT_N, elapsed, latencies, total_tokens)
    return [(label, r["text"]) for (label, _), (_, r) in zip(items, results)]


def run_vl_concurrent(engine):
    """N concurrent VL requests."""
    base = []
    for c in VL_CASES:
        uri = _image_uri(c["image"])
        prompt = _vl_prompt(c["question"], uri)
        base.append((c, prompt, uri))
    items = [base[i % len(base)] for i in range(CONCURRENT_N)]

    async def one(prompt, uri):
        t0 = time.perf_counter()
        r = await engine.async_generate(
            prompt=prompt, image_data=[uri], sampling_params=_VL_SAMPLING
        )
        return time.perf_counter() - t0, r

    async def run():
        t0 = time.perf_counter()
        results = await asyncio.gather(*(one(p, u) for _, p, u in items))
        return time.perf_counter() - t0, results

    elapsed, results = engine.loop.run_until_complete(run())
    latencies = [lat for lat, _ in results]
    total_tokens = sum(
        r.get("meta_info", {}).get("completion_tokens", 0) for _, r in results
    )

    seen = set()
    for (case, _, _), (_, r) in zip(items, results):
        name = case["image"]
        if name in seen:
            continue
        seen.add(name)
        print(f"  [{name}] {case['question']}\n    → {r['text']!r}")

    _report("vl-concurrent", CONCURRENT_N, elapsed, latencies, total_tokens)
    return [(case, r["text"]) for (case, _, _), (_, r) in zip(items, results)]


def run_mixed_concurrent(engine):
    """Mixed text + VL concurrent requests."""
    # Half text, half VL
    n_text = CONCURRENT_N // 2
    n_vl = CONCURRENT_N - n_text

    text_base = [(p, _text_prompt(p)) for p in TEXT_PROMPTS]
    text_items = [text_base[i % len(text_base)] for i in range(n_text)]

    vl_base = []
    for c in VL_CASES:
        uri = _image_uri(c["image"])
        prompt = _vl_prompt(c["question"], uri)
        vl_base.append((c, prompt, uri))
    vl_items = [vl_base[i % len(vl_base)] for i in range(n_vl)]

    async def text_one(prompt):
        t0 = time.perf_counter()
        r = await engine.async_generate(prompt=prompt, sampling_params=_TEXT_SAMPLING)
        return time.perf_counter() - t0, r, "text"

    async def vl_one(prompt, uri):
        t0 = time.perf_counter()
        r = await engine.async_generate(
            prompt=prompt, image_data=[uri], sampling_params=_VL_SAMPLING
        )
        return time.perf_counter() - t0, r, "vl"

    async def run():
        tasks = []
        for _, p in text_items:
            tasks.append(text_one(p))
        for _, p, u in vl_items:
            tasks.append(vl_one(p, u))
        t0 = time.perf_counter()
        results = await asyncio.gather(*tasks)
        return time.perf_counter() - t0, results

    elapsed, results = engine.loop.run_until_complete(run())
    latencies = [lat for lat, _, _ in results]
    total_tokens = sum(
        r.get("meta_info", {}).get("completion_tokens", 0) for _, r, _ in results
    )

    _report("mixed-concurrent", CONCURRENT_N, elapsed, latencies, total_tokens)

    # Split results back
    text_results = [
        (label, r["text"])
        for (label, _), (_, r, _) in zip(text_items, results[:n_text])
    ]
    vl_results = [
        (case, r["text"]) for (case, _, _), (_, r, _) in zip(vl_items, results[n_text:])
    ]
    return text_results, vl_results


# ─── Validation ──────────────────────────────────────────────────────────────


def validate_text(pairs):
    for prompt, text in pairs:
        assert len(text) > 0, f"Empty output for: {prompt!r}"
        if prompt in TEXT_EXPECTED:
            lower = text.lower()
            matched = any(pat in lower for pat in TEXT_EXPECTED[prompt])
            assert matched, (
                f"Expected one of {TEXT_EXPECTED[prompt]} in output for "
                f"{prompt!r}, got {text!r}"
            )
    print("  Text validation passed.")


def validate_vl(pairs):
    for case, text in pairs:
        assert len(text) > 0, f"Empty output for VL: {case['image']}"
        lower = text.lower()
        matched = any(pat.lower() in lower for pat in case["expected"])
        assert matched, (
            f"VL [{case['image']}]: expected one of {case['expected']}, got {text!r}"
        )
    print("  VL validation passed.")


# ─── Image check ─────────────────────────────────────────────────────────────


def _check_images():
    missing = [
        str(IMAGE_DIR / c["image"])
        for c in VL_CASES
        if not (IMAGE_DIR / c["image"]).is_file()
    ]
    if missing:
        print("ERROR: Missing test images:")
        for m in missing:
            print(f"  - {m}")
        print(f"\nRun: python {IMAGE_DIR / 'generate.py'}")
        print("And download cat.jpg (see generate.py docstring).")
        sys.exit(1)


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3.6-35B-A3B concurrent test")
    parser.add_argument(
        "--mode",
        choices=ALL_MODES + ["all"],
        default="all",
        help="Which concurrent mode to run (default: all)",
    )
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        print("Set MODEL_PATH to the correct path.")
        sys.exit(1)

    _check_images()

    print(
        f"Model: {MODEL_PATH} | TP: {TP_SIZE} | CONCURRENT_N: {CONCURRENT_N}\n"
        f"cuda_graph disabled — throughput numbers reflect that.\n"
    )

    engine = _make_engine()
    try:
        modes = ALL_MODES if args.mode == "all" else [args.mode]
        for i, m in enumerate(modes):
            if i > 0:
                engine.flush_cache()  # Release KV cache between modes to avoid OOM on VL prefill
            print(f"\n{'=' * 60}\n=== {m}\n{'=' * 60}")
            if m == "text":
                pairs = run_text_concurrent(engine)
                validate_text(pairs)
            elif m == "vl":
                pairs = run_vl_concurrent(engine)
                validate_vl(pairs)
            elif m == "mixed":
                t_pairs, v_pairs = run_mixed_concurrent(engine)
                validate_text(t_pairs)
                validate_vl(v_pairs)
    finally:
        engine.shutdown()

    print("\nAll concurrent tests passed.")
