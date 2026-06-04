# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-27B (Dense, hybrid attention) offline inference with sglang-plugin-FL.

This example verifies that the plugin can correctly serve a Qwen3.6 dense model
with hybrid attention (linear_attention + full_attention layers).

Usage:
  python qwen3_6_27b_offline_inference.py

Environment variables:
  MODEL_PATH    Model path (default: /models/Qwen3.6-27B)
  TP_SIZE       Tensor parallelism (default: 1)
  MAX_TOKENS    Max generation tokens (default: 10)
"""

import os
import sys

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", "/data/Qwen3.6-27B")
TP_SIZE = int(os.environ.get("TP_SIZE", "1"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "128"))

PROMPTS = [
    "How many states are there in the United States?",
    "The capital of France is",
]

EXPECTED_PARTS = {
    "The capital of France is": "Paris",
    "How many states are there in the United States?": "50",
}


# ─── Inference ────────────────────────────────────────────────────────────────


def run_engine():
    from sglang.srt.entrypoints.engine import Engine

    engine = Engine(
        model_path=MODEL_PATH,
        tp_size=TP_SIZE,
        mem_fraction_static=0.85,
        disable_cuda_graph=True,
        disable_piecewise_cuda_graph=True,
    )

    sampling_params = {"max_new_tokens": MAX_TOKENS, "temperature": 0}

    outputs = []
    for prompt in PROMPTS:
        result = engine.generate(prompt=prompt, sampling_params=sampling_params)
        text = result["text"]
        outputs.append(text)
        print(f"Prompt: {prompt!r}, Generated text: {text!r}")

    engine.shutdown()
    return outputs


# ─── Validation ───────────────────────────────────────────────────────────────


def validate(outputs):
    """Basic sanity checks on generated outputs."""
    assert len(outputs) == len(PROMPTS), (
        f"Expected {len(PROMPTS)} outputs, got {len(outputs)}"
    )

    for prompt, text in zip(PROMPTS, outputs):
        assert len(text) > 0, f"Empty output for prompt: {prompt!r}"
        if prompt in EXPECTED_PARTS:
            expected = EXPECTED_PARTS[prompt]
            # Strip thinking content if present (Qwen3 may emit <think>...</think>)
            answer = text
            if "</think>" in answer:
                answer = answer.split("</think>", 1)[1]
            assert expected in answer, (
                f"Expected {expected!r} in output for {prompt!r}, got {text!r}"
            )

    print("\n All validations passed.")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        print("Set MODEL_PATH to the correct path.")
        sys.exit(1)

    outputs = run_engine()
    validate(outputs)
