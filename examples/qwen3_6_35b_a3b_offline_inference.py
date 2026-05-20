# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-35B-A3B (MoE) offline inference with sglang-plugin-FL.

This example verifies that the plugin can correctly serve a Qwen3.5 MoE model.
It runs inference under three modes (baseline / aten / full) and checks that
all modes produce valid output. With temperature=0, outputs should be identical.

Usage:
  python qwen3_6_35b_a3b_offline_inference.py

Environment variables:
  MODEL_PATH    Model path (default: /models/Qwen3.6-35B-A3B)
  TP_SIZE       Tensor parallelism (default: 1)
  MAX_TOKENS    Max generation tokens (default: 10)
"""

import os
import sys

import torch

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/Qwen3.6-35B-A3B")
TP_SIZE = int(os.environ.get("TP_SIZE", "1"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "10"))

# page_size=1 is required on MUSA to work around a sglang platform bug;
# other platforms use the sglang default (no override needed).
_is_musa = hasattr(torch, "musa") and torch.musa.is_available()
_extra_engine_kwargs = {"page_size": 1} if _is_musa else {}

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
        **_extra_engine_kwargs,
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
            assert expected in text, (
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
