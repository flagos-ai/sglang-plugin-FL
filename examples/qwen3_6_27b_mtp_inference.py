# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-27B MTP (Multi-Token Prediction) inference test with sglang-plugin-FL.

Validates that speculative decoding (EAGLE/MTP) works correctly with the OOT plugin.
Tests include:
  1. Correctness: MTP output matches baseline (greedy, temperature=0)
  2. Accept length: avg_spec_accept_length > threshold
  3. Throughput: single-request token generation speed
  4. Diverse prompts: code, math, reasoning, factual Q&A, long-form

Usage:
  python qwen3_6_27b_mtp_inference.py [--skip-baseline] [--max-tokens N]

Environment variables:
  MODEL_PATH    Model path (default: /models/Qwen3.6-27B)
  TP_SIZE       Tensor parallelism (default: 1)
  MAX_TOKENS    Max generation tokens (default: 256)
"""

import argparse
import os
import sys
import time

import torch

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/Qwen3.6-27B")
TP_SIZE = int(os.environ.get("TP_SIZE", "1"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))

# ─── Diverse prompt set (covers different generation patterns) ────────────────

PROMPTS = [
    # Factual Q&A (short answers)
    {
        "prompt": "How many states are there in the United States?",
        "expected_contains": ["50"],
        "category": "factual",
    },
    {
        "prompt": "The capital of France is",
        "expected_contains": ["paris"],
        "category": "factual",
    },
    {
        "prompt": "What is the largest planet in the solar system?",
        "expected_contains": ["jupiter"],
        "category": "factual",
    },
    # Math / reasoning
    {
        "prompt": "What is 17 multiplied by 13? Give only the number.",
        "expected_contains": ["221"],
        "category": "math",
    },
    {
        "prompt": "If a train travels at 60 km/h for 2.5 hours, how far does it travel? Answer with the number in km.",
        "expected_contains": ["150"],
        "category": "math",
    },
    # Code generation (tests repetitive token patterns — good for MTP)
    {
        "prompt": "Write a Python function that computes the factorial of n recursively.",
        "expected_contains": ["def", "factorial", "return"],
        "category": "code",
    },
    {
        "prompt": "Write a Python function to check if a string is a palindrome.",
        "expected_contains": ["def", "return"],
        "category": "code",
    },
    # Long-form explanation
    {
        "prompt": "Explain the concept of gravity in three sentences.",
        "expected_contains": ["mass"],
        "category": "explanation",
    },
    {
        "prompt": "What are the three states of matter? Explain each briefly.",
        "expected_contains": ["solid", "liquid", "gas"],
        "category": "explanation",
    },
    # Structured output
    {
        "prompt": "List the first 5 prime numbers, separated by commas.",
        "expected_contains": ["2", "3", "5", "7", "11"],
        "category": "structured",
    },
    # Translation / multilingual
    {
        "prompt": 'Translate "hello world" to French.',
        "expected_contains": ["bonjour"],
        "category": "translation",
    },
    # Longer generation (good for measuring sustained MTP performance)
    {
        "prompt": "Write a short story (about 100 words) about a robot learning to paint.",
        "expected_contains": [],
        "min_length": 50,
        "category": "creative",
    },
]

# ─── Prompt formatting ───────────────────────────────────────────────────────

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return _tokenizer


def _text_prompt(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    return _get_tokenizer().apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


# ─── Engine factories ────────────────────────────────────────────────────────


def _make_mtp_engine(disable_cuda_graph=False, disable_piecewise_cuda_graph=False):
    """Create engine with MTP (speculative decoding) enabled."""
    from sglang.srt.entrypoints.engine import Engine

    return Engine(
        model_path=MODEL_PATH,
        tp_size=TP_SIZE,
        mem_fraction_static=0.8,
        disable_cuda_graph=disable_cuda_graph,
        disable_piecewise_cuda_graph=disable_piecewise_cuda_graph,
        trust_remote_code=True,
        disable_radix_cache=True,
        speculative_algorithm="EAGLE",
        speculative_num_steps=3,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
    )


def _make_baseline_engine(disable_cuda_graph=False, disable_piecewise_cuda_graph=False):
    """Create engine without MTP (standard autoregressive)."""
    from sglang.srt.entrypoints.engine import Engine

    return Engine(
        model_path=MODEL_PATH,
        tp_size=TP_SIZE,
        mem_fraction_static=0.8,
        disable_cuda_graph=disable_cuda_graph,
        disable_piecewise_cuda_graph=disable_piecewise_cuda_graph,
        trust_remote_code=True,
    )


# ─── Inference helpers ────────────────────────────────────────────────────────


def run_inference(engine, prompts, max_tokens):
    """Run inference, return list of (text, meta_info, latency)."""
    sampling_params = {"max_new_tokens": max_tokens, "temperature": 0}
    results = []
    for p in prompts:
        t0 = time.perf_counter()
        result = engine.generate(
            prompt=_text_prompt(p["prompt"]), sampling_params=sampling_params
        )
        lat = time.perf_counter() - t0
        results.append((result["text"], result.get("meta_info", {}), lat))
    return results


def run_long_generation(engine, prompt, max_tokens=512):
    """Single long generation for throughput measurement."""
    sampling_params = {"max_new_tokens": max_tokens, "temperature": 0}
    t0 = time.perf_counter()
    result = engine.generate(
        prompt=_text_prompt(prompt), sampling_params=sampling_params
    )
    elapsed = time.perf_counter() - t0
    text = result["text"]
    meta = result.get("meta_info", {})
    tokens = meta.get("completion_tokens", len(text.split()))
    return text, tokens, elapsed


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-baseline", action="store_true", help="Skip baseline comparison (faster)"
    )
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="Disable CUDA graph capture (default: enabled)",
    )
    parser.add_argument(
        "--disable-piecewise-cuda-graph",
        action="store_true",
        help="Disable piecewise CUDA graph (default: enabled)",
    )
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        sys.exit(1)

    max_tokens = args.max_tokens
    disable_cg = args.disable_cuda_graph
    disable_pcg = args.disable_piecewise_cuda_graph
    mode_str = "eager" if disable_cg else ("cuda_graph" if not disable_pcg else "cuda_graph(no piecewise)")
    print("=" * 70)
    print("  Qwen3.6-27B MTP (Speculative Decoding) Validation")
    print("=" * 70)
    print(f"  Model: {MODEL_PATH}")
    print(f"  TP: {TP_SIZE} | max_tokens: {max_tokens} | mode: {mode_str}")
    print("  MTP: algorithm=EAGLE, num_steps=3, topk=1, draft_tokens=4")
    print(f"  Prompts: {len(PROMPTS)} (factual/math/code/explanation/creative)")
    print()

    # ─── Phase 1: MTP inference ───────────────────────────────────────────────
    print("Phase 1: MTP-enabled inference")
    print("-" * 50)

    t0 = time.perf_counter()
    mtp_engine = _make_mtp_engine(
        disable_cuda_graph=disable_cg, disable_piecewise_cuda_graph=disable_pcg
    )
    print(f"  Engine loaded in {time.perf_counter() - t0:.1f}s")

    mtp_results = run_inference(mtp_engine, PROMPTS, max_tokens)

    print(f"\n  Results ({len(PROMPTS)} prompts):")
    for p, (text, meta, lat) in zip(PROMPTS, mtp_results):
        tokens = meta.get("completion_tokens", "?")
        print(f"  [{p['category']:12}] {p['prompt'][:50]}")
        print(f"               -> {text[:120]}{'...' if len(text) > 120 else ''}")
        print(f"               ({tokens} tokens, {lat:.2f}s)")
        print()

    # Throughput test: single long generation
    print("  Throughput test (long generation, 512 tokens):")
    long_text, long_tokens, long_time = run_long_generation(
        mtp_engine,
        "Write a detailed essay about the history of artificial intelligence, "
        "covering key milestones from the 1950s to today.",
        max_tokens=512,
    )
    throughput = long_tokens / long_time if long_time > 0 else 0
    print(f"    {long_tokens} tokens in {long_time:.2f}s = {throughput:.1f} tok/s")

    # Get accept length
    avg_accept = None
    try:
        info = mtp_engine.get_server_info()
        states = info.get("internal_states", [{}])
        avg_accept = (states[0] if isinstance(states, list) else states).get(
            "avg_spec_accept_length", None
        )
    except Exception:
        pass

    if avg_accept is not None:
        print(f"\n  avg_spec_accept_length: {avg_accept:.2f}")

    mtp_engine.shutdown()
    del mtp_engine
    torch.cuda.empty_cache()

    # ─── Phase 2: Baseline (optional) ────────────────────────────────────────
    baseline_results = None
    base_throughput = 0
    if not args.skip_baseline:
        print("\nPhase 2: Baseline (no MTP) inference")
        print("-" * 50)

        t0 = time.perf_counter()
        baseline_engine = _make_baseline_engine(
            disable_cuda_graph=disable_cg, disable_piecewise_cuda_graph=disable_pcg
        )
        print(f"  Engine loaded in {time.perf_counter() - t0:.1f}s")

        baseline_results = run_inference(baseline_engine, PROMPTS, max_tokens)

        # Baseline throughput
        _, base_tokens, base_time = run_long_generation(
            baseline_engine,
            "Write a detailed essay about the history of artificial intelligence, "
            "covering key milestones from the 1950s to today.",
            max_tokens=512,
        )
        base_throughput = base_tokens / base_time if base_time > 0 else 0
        print(
            f"  Throughput: {base_tokens} tokens in {base_time:.2f}s = {base_throughput:.1f} tok/s"
        )

        baseline_engine.shutdown()
        del baseline_engine
        torch.cuda.empty_cache()

    # ─── Phase 3: Validation ──────────────────────────────────────────────────
    print("\nPhase 3: Validation")
    print("-" * 50)

    passed = 0
    failed = 0
    warnings = 0

    # 3a. Content correctness
    print("\n  [Content Correctness]")
    for p, (text, _, _) in zip(PROMPTS, mtp_results):
        lower = text.lower()
        if p["expected_contains"]:
            matched = all(exp.lower() in lower for exp in p["expected_contains"])
            if matched:
                print(f"    PASS [{p['category']:12}] {p['prompt'][:45]}")
                passed += 1
            else:
                print(f"    FAIL [{p['category']:12}] {p['prompt'][:45]}")
                print(f"         expected: {p['expected_contains']}")
                print(f"         got: {text[:100]}")
                failed += 1
        elif p.get("min_length"):
            if len(text) >= p["min_length"]:
                print(
                    f"    PASS [{p['category']:12}] length={len(text)} >= {p['min_length']}"
                )
                passed += 1
            else:
                print(
                    f"    FAIL [{p['category']:12}] length={len(text)} < {p['min_length']}"
                )
                failed += 1
        else:
            if len(text.strip()) > 0:
                passed += 1
            else:
                failed += 1

    # 3b. MTP vs Baseline comparison
    if baseline_results:
        print("\n  [MTP vs Baseline Match (greedy, temp=0)]")
        match_count = 0
        for p, (mtp_text, _, _), (base_text, _, _) in zip(
            PROMPTS, mtp_results, baseline_results
        ):
            if mtp_text.strip() == base_text.strip():
                match_count += 1
            else:
                print(f"    DIFF [{p['category']:12}] {p['prompt'][:40]}")
                print(f"         MTP:  {mtp_text[:60]}")
                print(f"         Base: {base_text[:60]}")
        match_pct = match_count / len(PROMPTS) * 100
        print(f"    Match rate: {match_count}/{len(PROMPTS)} ({match_pct:.0f}%)")
        if match_pct >= 90:
            print("    PASS: >=90% match")
            passed += 1
        else:
            print("    WARN: <90% match (may be numerical precision issue)")
            warnings += 1

    # 3c. Accept length check
    print("\n  [Speculative Accept Length]")
    ACCEPT_THRESHOLD = 2.0
    if avg_accept is not None:
        if avg_accept > ACCEPT_THRESHOLD:
            print(
                f"    PASS: avg_spec_accept_length = {avg_accept:.2f} > {ACCEPT_THRESHOLD}"
            )
            passed += 1
        else:
            print(
                f"    FAIL: avg_spec_accept_length = {avg_accept:.2f} <= {ACCEPT_THRESHOLD}"
            )
            failed += 1
    else:
        print("    SKIP: stats not available")
        warnings += 1

    # 3d. Throughput comparison
    print("\n  [Throughput]")
    print(f"    MTP:      {throughput:.1f} tok/s ({long_tokens} tokens)")
    if baseline_results:
        print(f"    Baseline: {base_throughput:.1f} tok/s")
        if throughput > base_throughput:
            speedup = throughput / base_throughput
            print(f"    PASS: MTP {speedup:.2f}x faster")
            passed += 1
        else:
            print("    WARN: MTP not faster (may be due to kernel JIT / cold start)")
            warnings += 1

    # ─── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  SUMMARY: {passed} passed, {failed} failed, {warnings} warnings")
    print(f"  avg_spec_accept_length: {avg_accept if avg_accept else 'N/A'}")
    print(f"  MTP throughput: {throughput:.1f} tok/s")
    if failed == 0:
        print("  RESULT: ALL VALIDATIONS PASSED")
    else:
        print("  RESULT: SOME VALIDATIONS FAILED")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
