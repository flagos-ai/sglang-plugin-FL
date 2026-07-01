# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-35B-A3B (MoE) offline inference with sglang-plugin-FL.

Supports CUDA, MUSA, Ascend NPU, and TsingMicro txda; platform-specific
settings are applied automatically at runtime.

Usage:
  python qwen3_6_35b_a3b_offline_inference.py

Environment variables:
  MODEL_PATH    Model path (default: /models/Qwen3.6-35B-A3B)
  TP_SIZE       Tensor parallelism (default: 1)
  MAX_TOKENS    Max generation tokens (default: 10)
  IMAGE_DIR     Test image directory (default: examples/test_images/ next to this file)
"""

import os
import sys
import types
from pathlib import Path
try:
    import torch_txda  # noqa: F401
    from torch_txda import transfer_to_txda
except ImportError:
    pass
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
    # os.environ.setdefault("SGLANG_ENABLE_OVERLAP_PLAN_STREAM", "0")
    # os.environ.setdefault("TCCL_BUFFSIZE", "2400")
    os.environ.setdefault("SGLANG_FL_TIMER_ENABLE", "1")

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/Qwen3.6-35B-A3B")
TP_SIZE = int(os.environ.get("TP_SIZE", "8" if _is_txda else ("4" if _is_npu else "1")))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "10"))

_HERE = Path(__file__).resolve().parent
IMAGE_DIR = Path(os.environ.get("IMAGE_DIR", _HERE / "test_images"))

# page_size=1 is required on MUSA to work around a sglang platform bug.
# Ascend NPU requires its own attention backend and extra runtime settings.
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
    print("inference use txda")
    # ─── Early stub-module injection ─────────────────────────────────────────────
    # Must run at *import time*, before any SGLang submodule triggers a cascade
    # of real imports of sgl_kernel / flashinfer / pynccl_allocator.  The
    # patches subsystem's load_plugin() entry point runs too late for that.
    try:
        from sglang_fl.dispatch.backends.vendor.txda.patches.platform_stubs import patch as _patch_stubs
        _patch_stubs()
    except Exception:
        pass  # best-effort: if this fails, the downstream import chain will show the real error
    _extra_engine_kwargs = {
        "device": "txda",
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "disable_radix_cache": True,
        "watchdog_timeout": 3600,
        "mm_attention_backend": "triton_attn",
        "disable_fast_image_processor": True,
        "mem_fraction_static": 0.6,
        "context_length": 8192,
        "chunked_prefill_size":256
    }
else:
    _extra_engine_kwargs = {"trust_remote_code": True}

TEXT_PROMPTS = [
    "How many states are there in the United States?",
    "The capital of France is",
]

TEXT_EXPECTED = {
    "The capital of France is": "Paris",
    "How many states are there in the United States?": "50",
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


# ─── Inference ────────────────────────────────────────────────────────────────


def run_engine():

    from sglang.srt.entrypoints.engine import Engine

    engine = Engine(
        model_path=MODEL_PATH,
        tp_size=TP_SIZE,
        # mem_fraction_static=0.85,
        disable_cuda_graph=True,
        disable_piecewise_cuda_graph=True,
        **_extra_engine_kwargs,
    )

    sampling_params = {"max_new_tokens": MAX_TOKENS, "temperature": 0}
    vl_sampling = {"max_new_tokens": MAX_TOKENS, "temperature": 0}

    # --- Text inference ---
    print("=== Text Inference ===")
    text_outputs = []
    for prompt in TEXT_PROMPTS:
        result = engine.generate(
            prompt=_text_prompt(prompt), sampling_params=sampling_params
        )
        text = result["text"]
        text_outputs.append(text)
        print(f"  Prompt: {prompt!r}\n    → {text!r}")

    # --- Multimodal inference ---
    print("\n=== Multimodal Inference ===")
    vl_outputs = []
    for case in VL_CASES:
        img_path = IMAGE_DIR / case["image"]
        if not img_path.is_file():
            print(f"  SKIP (missing image): {img_path}")
            vl_outputs.append(None)
            continue
        uri = _image_uri(case["image"])
        result = engine.generate(
            prompt=_vl_prompt(case["question"], uri),
            image_data=[uri],
            sampling_params=vl_sampling,
        )
        text = result["text"]
        vl_outputs.append(text)
        print(f"  Prompt: [{case['image']}] {case['question']}\n    → {text!r}")


    engine.shutdown()
    return text_outputs, vl_outputs


# ─── Validation ───────────────────────────────────────────────────────────────


def validate(text_outputs, vl_outputs):
    """Basic sanity checks on generated outputs."""
    # Text validation
    assert len(text_outputs) == len(TEXT_PROMPTS)
    for prompt, text in zip(TEXT_PROMPTS, text_outputs):
        assert len(text) > 0, f"Empty output for prompt: {prompt!r}"
        if prompt in TEXT_EXPECTED:
            expected = TEXT_EXPECTED[prompt]
            assert expected in text, (
                f"Expected {expected!r} in output for {prompt!r}, got {text!r}"
            )

    # VL validation
    for case, text in zip(VL_CASES, vl_outputs):
        if text is None:
            continue
        assert len(text) > 0, f"Empty output for VL case: {case['image']}"
        lower = text.lower()
        matched = any(pat.lower() in lower for pat in case["expected"])
        assert matched, (
            f"VL case [{case['image']}]: expected one of {case['expected']} "
            f"in output, got {text!r}"
        )

    print("\nAll validations passed.")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        print("Set MODEL_PATH to the correct path.")
        sys.exit(1)

    text_outputs, vl_outputs = run_engine()
    validate(text_outputs, vl_outputs)
