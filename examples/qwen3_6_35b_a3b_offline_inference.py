# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-35B-A3B (MoE) offline inference with sglang-plugin-FL.

Supports CUDA, MUSA, and Ascend NPU; platform-specific settings are applied
automatically at runtime.

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

# ─── Platform detection ───────────────────────────────────────────────────────

_is_musa = hasattr(torch, "musa") and torch.musa.is_available()
_is_npu = hasattr(torch, "npu") and torch.npu.is_available()

# Must be set before importing sglang.
if _is_npu:
    os.environ.setdefault("SGLANG_ENABLE_OVERLAP_PLAN_STREAM", "0")
    os.environ.setdefault("SGLANG_ENABLE_SPEC_V2", "1")
    os.environ.setdefault("HCCL_BUFFSIZE", "2400")
    os.environ.setdefault("SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "128")

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/Qwen3.6-35B-A3B")
TP_SIZE = int(os.environ.get("TP_SIZE", "2" if _is_npu else "1"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "10"))

# page_size=1 is required on MUSA to work around a sglang platform bug.
# Ascend NPU requires its own attention backend and extra runtime settings.
if _is_musa:
    _extra_engine_kwargs: dict = {"page_size": 1}
elif _is_npu:
    _extra_engine_kwargs = {
        "attention_backend": "ascend",
        "device": "npu",
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "disable_radix_cache": True,
    }
else:
    _extra_engine_kwargs = {}

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
