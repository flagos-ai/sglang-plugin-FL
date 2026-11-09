# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Xing4.0-29B-A4B offline inference with sglang-plugin-FL.

Xing4.0 is a 29B total / 4B active MoE model (MLA attention + mHC residual
streams). The mHC pre/post pair is the part this plugin owns: on Ascend NPU it
is dispatched through the FlagGems backend, on other platforms SGLang's native
TileLang kernel is used.

The model implementation ships inside the checkpoint (``trust_remote_code``),
so the plugin only has to provide the operator backend -- no model code lives in
this repository.

Usage:
  python xing4_0_29b_a4b_offline_inference.py

Environment variables:
  MODEL_PATH    Model path (default: /models/Xing4.0-29B-A4B)
  TP_SIZE       Tensor parallelism (default: 4 on NPU)
  MAX_TOKENS    Max generation tokens (default: 32)
"""

import os
import sys

import torch

# ─── Platform detection ───────────────────────────────────────────────────────

_is_npu = hasattr(torch, "npu") and torch.npu.is_available()

# Must be set before importing sglang.
if _is_npu:
    os.environ.setdefault("SGLANG_ENABLE_OVERLAP_PLAN_STREAM", "0")
    os.environ.setdefault("HCCL_BUFFSIZE", "2400")

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/Xing4.0-29B-A4B")
TP_SIZE = int(os.environ.get("TP_SIZE", "4" if _is_npu else "1"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "32"))
MODEL_NAME = "Xing4.0-29B-A4B"

PROMPTS = [
    "What is the capital of France? Answer with one word.",
    "How many states are there in the United States?",
]

EXPECTED = {
    "What is the capital of France? Answer with one word.": "Paris",
    "How many states are there in the United States?": "50",
}

if _is_npu:
    _extra_engine_kwargs: dict = {
        "attention_backend": "ascend",
        "device": "npu",
        "dtype": "bfloat16",
        # SGLang's piecewise CUDA graph path is incompatible with FlagGems
        # Triton kernels, which emit logging calls torch.compile cannot trace.
        "disable_piecewise_cuda_graph": True,
        # Xing4 routes mHC through the plugin; shared-expert fusion is not
        # implemented for this architecture.
        "disable_shared_experts_fusion": True,
    }
else:
    _extra_engine_kwargs = {"device": "cuda", "dtype": "bfloat16"}


# ─── Engine ──────────────────────────────────────────────────────────────────


def _chat_prompt(prompt: str) -> str:
    # Xing4 ships its own chat template (`<_system>` marker, built-in system
    # prompt), so build the prompt through the tokenizer rather than hardcoding
    # a template that would silently drift from the checkpoint.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def run_engine():
    import sglang as sgl

    engine = sgl.Engine(
        model_path=MODEL_PATH,
        served_model_name=MODEL_NAME,
        trust_remote_code=True,
        tp_size=TP_SIZE,
        mem_fraction_static=0.80,
        **_extra_engine_kwargs,
    )

    sampling_params = {"max_new_tokens": MAX_TOKENS, "temperature": 0}

    outputs = []
    for prompt in PROMPTS:
        result = engine.generate(
            prompt=_chat_prompt(prompt), sampling_params=sampling_params
        )
        text = result["text"]
        outputs.append(text)
        print(f"  Prompt: {prompt!r}\n    -> {text!r}")

    engine.shutdown()
    return outputs


# ─── Validation ───────────────────────────────────────────────────────────────


def validate(outputs):
    assert len(outputs) == len(PROMPTS), "missing outputs"
    for prompt, text in zip(PROMPTS, outputs):
        assert len(text) > 0, f"Empty output for prompt: {prompt!r}"
        expected = EXPECTED.get(prompt)
        if expected:
            assert expected.lower() in text.lower(), (
                f"Expected {expected!r} in output for {prompt!r}, got {text!r}"
            )
    print("\nAll validations passed.")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        print("Set MODEL_PATH to the correct path.")
        sys.exit(1)

    outputs = run_engine()
    validate(outputs)
