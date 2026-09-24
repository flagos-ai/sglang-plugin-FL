# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-27B MTP (Multi-Token Prediction) inference test with sglang-plugin-FL.

Validates that speculative decoding (EAGLE/MTP) works correctly with the OOT plugin.
Tests include:
  1. Correctness: MTP and baseline outputs satisfy the same semantic contracts
  2. Accept length: avg_spec_accept_length > threshold
  3. Throughput: single-request token generation speed
  4. Diverse prompts: code, math, reasoning, factual Q&A, long-form

Usage:
  python qwen3_6_27b_mtp_inference.py [--skip-baseline] [--max-tokens N]

On Ascend, the validated MTP path always uses eager, synchronous execution. The
script therefore forces CUDA graph, piecewise CUDA graph, and overlap scheduling
off even when the corresponding command-line switches are omitted.

Environment variables:
  MODEL_PATH    Model path (default: /models/Qwen3.6-27B)
  TP_SIZE       Tensor parallelism (default: 4 on Ascend/TXDA, otherwise 1)
  MAX_TOKENS    Max generation tokens (default: 256)
"""

import argparse
import ast
import inspect
import math
import os
import re
import sys
import time

import torch

# ─── Platform detection ───────────────────────────────────────────────────────

_is_musa = hasattr(torch, "musa") and torch.musa.is_available()
_is_npu = hasattr(torch, "npu") and torch.npu.is_available()
_is_txda = hasattr(torch, "txda") and torch.txda.is_available()
_is_corex = hasattr(torch, "corex") and torch.cuda.is_available()
_is_hcu = hasattr(torch, "__hcu_version__") and torch.cuda.is_available()

# These settings must be present before importing sglang.
if _is_npu:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    if not any(
        name in os.environ
        for name in (
            "HCCL_HOST_SOCKET_PORT_RANGE",
            "HCCL_NPU_SOCKET_PORT_RANGE",
            "HCCL_IF_BASE_PORT",
        )
    ):
        os.environ["HCCL_HOST_SOCKET_PORT_RANGE"] = "auto"
        os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = "auto"
    os.environ.setdefault("SGLANG_ENABLE_OVERLAP_PLAN_STREAM", "0")
    os.environ.setdefault("HCCL_BUFFSIZE", "2400")
    os.environ.setdefault("SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "128")

if _is_txda:
    os.environ.setdefault("SGLANG_FL_TIMER_ENABLE", "1")
    os.environ.setdefault("SGLANG_REQ_WAITING_TIMEOUT", "-1")
    os.environ.setdefault("SGLANG_REQ_RUNNING_TIMEOUT", "-1")

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/Qwen3.6-27B")
TP_SIZE = int(os.environ.get("TP_SIZE", "4" if _is_npu or _is_txda else "1"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))

# ─── Diverse prompt set (covers different generation patterns) ────────────────

PROMPTS = [
    # Factual Q&A (short answers)
    {
        "prompt": "How many states are there in the United States? Give only the integer.",
        "expected_number": 50,
        "category": "factual",
    },
    {
        "prompt": "What is the capital of France? Give only the city name.",
        "expected_exact": ["paris"],
        "category": "factual",
    },
    {
        "prompt": (
            "What is the largest planet in the solar system? Give only the planet name."
        ),
        "expected_exact": ["jupiter"],
        "category": "factual",
    },
    # Math / reasoning
    {
        "prompt": "What is 17 multiplied by 13? Give only the number.",
        "expected_number": 221,
        "category": "math",
    },
    {
        "prompt": (
            "If a train travels at 60 km/h for 2.5 hours, how far does it travel? "
            "Give only the number, without a unit."
        ),
        "expected_number": 150,
        "category": "math",
    },
    # Code generation (tests repetitive token patterns — good for MTP)
    {
        "prompt": (
            "Write only a concise Python function named factorial that computes the "
            "factorial of n recursively. Do not add a docstring, comments, or explanation."
        ),
        "python_function": "factorial",
        "python_contract": "recursive",
        "category": "code",
    },
    {
        "prompt": (
            "Write only a concise Python function named is_palindrome that checks whether "
            "a string is a palindrome by comparing it with its [::-1] slice. Do not add "
            "a docstring, comments, or explanation."
        ),
        "python_function": "is_palindrome",
        "python_contract": "reverse_slice_comparison",
        "category": "code",
    },
    # Long-form explanation
    {
        "prompt": "Explain the concept of gravity in three sentences.",
        "expected_terms": ["mass"],
        "category": "explanation",
    },
    {
        "prompt": "What are the three states of matter? Explain each briefly.",
        "expected_terms": ["solid", "liquid", "gas"],
        "category": "explanation",
    },
    # Structured output
    {
        "prompt": (
            "Give only the first 5 prime numbers separated by commas, with no other text."
        ),
        "expected_number_sequence": [2, 3, 5, 7, 11],
        "category": "structured",
    },
    # Translation / multilingual
    {
        "prompt": (
            'Translate "hello world" to French. Give only the French translation.'
        ),
        "expected_exact": ["bonjour le monde", "bonjour tout le monde"],
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


def _piecewise_graph_kwargs(disabled: bool):
    """Map the prefill graph switch across SGLang's old and new APIs."""
    if not disabled:
        return {}

    from sglang.srt.server_args import ServerArgs

    parameters = inspect.signature(ServerArgs).parameters
    if "disable_prefill_cuda_graph" in parameters:
        return {"disable_prefill_cuda_graph": True}
    return {"disable_piecewise_cuda_graph": True}


def _platform_engine_kwargs() -> dict:
    """Return device-specific Engine arguments without CUDA assumptions."""
    if _is_musa:
        return {"page_size": 1}
    if _is_npu:
        return {
            "attention_backend": "ascend",
            "device": "npu",
            "dtype": "bfloat16",
        }
    if _is_txda:
        try:
            from sglang_fl.dispatch.backends.vendor.tsingmicro.patches.platform_stubs import (
                patch as patch_platform_stubs,
            )

            patch_platform_stubs()
        except Exception:
            pass
        return {
            "device": "txda",
            "dtype": "bfloat16",
            "watchdog_timeout": 3600,
            "mm_attention_backend": "triton_attn",
            "disable_fast_image_processor": True,
            "context_length": 8192,
            "chunked_prefill_size": 256,
        }
    if _is_corex:
        return {
            "attention_backend": "triton",
            "watchdog_timeout": 3600,
            "cuda_graph_max_bs": 16,
        }
    if _is_hcu:
        return {
            "dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "page_size": 64,
            "enable_breakable_cuda_graph": False,
        }
    return {}


def _empty_device_cache() -> None:
    """Release cached accelerator memory using the active torch backend."""
    for backend_name in ("npu", "musa", "txda", "cuda"):
        backend = getattr(torch, backend_name, None)
        if backend is None or not hasattr(backend, "empty_cache"):
            continue
        is_available = getattr(backend, "is_available", None)
        if callable(is_available) and not is_available():
            continue
        backend.empty_cache()
        return


def _enforce_ascend_mtp_runtime_mode(args) -> None:
    """Keep Ascend MTP inside the eager mode covered by the correctness gate."""
    if not _is_npu:
        return

    if not (
        args.disable_cuda_graph
        and args.disable_piecewise_cuda_graph
        and args.disable_overlap_schedule
    ):
        print(
            "Ascend MTP requires the validated eager runtime mode; forcing "
            "CUDA graphs and overlap scheduling off."
        )
    args.disable_cuda_graph = True
    args.disable_piecewise_cuda_graph = True
    args.disable_overlap_schedule = True


def _make_mtp_engine(
    disable_cuda_graph=False,
    disable_piecewise_cuda_graph=False,
    disable_overlap_schedule=False,
):
    """Create engine with MTP (speculative decoding) enabled."""
    from sglang.srt.entrypoints.engine import Engine

    return Engine(
        model_path=MODEL_PATH,
        tp_size=TP_SIZE,
        mem_fraction_static=0.8,
        disable_cuda_graph=disable_cuda_graph,
        disable_overlap_schedule=disable_overlap_schedule,
        trust_remote_code=True,
        disable_radix_cache=True,
        speculative_algorithm="EAGLE",
        speculative_num_steps=3,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
        **_platform_engine_kwargs(),
        **_piecewise_graph_kwargs(disable_piecewise_cuda_graph),
    )


def _make_baseline_engine(
    disable_cuda_graph=False,
    disable_piecewise_cuda_graph=False,
    disable_overlap_schedule=False,
):
    """Create engine without MTP (standard autoregressive)."""
    from sglang.srt.entrypoints.engine import Engine

    return Engine(
        model_path=MODEL_PATH,
        tp_size=TP_SIZE,
        mem_fraction_static=0.8,
        disable_cuda_graph=disable_cuda_graph,
        disable_overlap_schedule=disable_overlap_schedule,
        trust_remote_code=True,
        # MTP uses fresh prefixes. Reusing hybrid states only in the baseline
        # changes prefill shapes and rounding, confounding greedy comparison.
        disable_radix_cache=True,
        **_platform_engine_kwargs(),
        **_piecewise_graph_kwargs(disable_piecewise_cuda_graph),
    )


# ─── Inference helpers ────────────────────────────────────────────────────────


def run_inference(engine, prompts, max_tokens):
    """Run inference, return list of (text, meta_info, latency)."""
    sampling_params = {"max_new_tokens": max_tokens, "temperature": 0}
    results = []
    for index, p in enumerate(prompts, start=1):
        t0 = time.perf_counter()
        result = engine.generate(
            prompt=_text_prompt(p["prompt"]), sampling_params=sampling_params
        )
        lat = time.perf_counter() - t0
        results.append((result["text"], result.get("meta_info", {}), lat))
        print(
            f"  Completed {index}/{len(prompts)} [{p['category']}]: "
            f"{result.get('meta_info', {}).get('completion_tokens', '?')} tokens "
            f"in {lat:.2f}s",
            flush=True,
        )
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


def _extract_python_source(text: str) -> str:
    """Remove one Markdown code fence without accepting surrounding prose."""
    source = text.strip()
    if not source.startswith("```"):
        return source

    lines = source.splitlines()
    if not lines or not lines[0].startswith("```"):
        return source
    lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines.pop()
    return "\n".join(lines).strip()


_SAFE_PYTHON_NODES = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.If,
    ast.IfExp,
    ast.Compare,
    ast.BinOp,
    ast.BoolOp,
    ast.UnaryOp,
    ast.Subscript,
    ast.Slice,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Call,
    ast.Mult,
    ast.Sub,
    ast.Add,
    ast.Mod,
    ast.FloorDiv,
    ast.And,
    ast.Or,
    ast.Not,
    ast.USub,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


def _run_restricted_python_contract(tree, function_name: str, contract: str):
    """Execute a single tiny function after rejecting side-effecting syntax."""
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return False, "expected exactly one synchronous function"
    function_node = tree.body[0]
    if function_node.name != function_name or function_node.decorator_list:
        return False, f"unexpected function definition for {function_name}"
    args = function_node.args
    if (
        len(args.posonlyargs) + len(args.args) != 1
        or args.vararg is not None
        or args.kwarg is not None
        or args.kwonlyargs
        or args.defaults
        or args.kw_defaults
    ):
        return False, f"function {function_name} must accept exactly one argument"

    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_PYTHON_NODES):
            return False, f"unsafe or unsupported Python node: {type(node).__name__}"
        if isinstance(node, ast.Call) and not (
            isinstance(node.func, ast.Name) and node.func.id == function_name
        ):
            return False, "only direct recursion is allowed"

    argument_name = (args.posonlyargs + args.args)[0].arg
    if contract == "recursive":
        has_branch = any(
            isinstance(node, (ast.If, ast.IfExp)) for node in ast.walk(function_node)
        )
        has_decrementing_recursive_call = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == function_name
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.BinOp)
            and isinstance(node.args[0].left, ast.Name)
            and node.args[0].left.id == argument_name
            and isinstance(node.args[0].op, ast.Sub)
            and isinstance(node.args[0].right, ast.Constant)
            and node.args[0].right.value == 1
            for node in ast.walk(function_node)
        )
        if not has_branch or not has_decrementing_recursive_call:
            return (
                False,
                "factorial contract requires a branch and direct n - 1 recursion",
            )
    elif contract == "reverse_slice_comparison":

        def is_reverse_slice(node):
            return (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == argument_name
                and isinstance(node.slice, ast.Slice)
                and node.slice.lower is None
                and node.slice.upper is None
                and isinstance(node.slice.step, ast.UnaryOp)
                and isinstance(node.slice.step.op, ast.USub)
                and isinstance(node.slice.step.operand, ast.Constant)
                and node.slice.step.operand.value == 1
            )

        has_reverse_slice_comparison = any(
            isinstance(node, ast.Compare)
            and (
                (
                    isinstance(node.left, ast.Name)
                    and node.left.id == argument_name
                    and any(is_reverse_slice(item) for item in node.comparators)
                )
                or (
                    is_reverse_slice(node.left)
                    and any(
                        isinstance(item, ast.Name) and item.id == argument_name
                        for item in node.comparators
                    )
                )
            )
            for node in ast.walk(function_node)
        )
        if not has_reverse_slice_comparison:
            return (
                False,
                "palindrome contract requires comparison with the s[::-1] slice",
            )

    try:
        namespace = {"__builtins__": {}, "bool": bool, "int": int, "str": str}
        exec(compile(tree, "<generated-contract>", "exec"), namespace)
        function = namespace[function_name]
        if contract == "recursive":
            cases = [(0, 1), (1, 1), (5, 120)]
        elif contract == "reverse_slice_comparison":
            cases = [("", True), ("abba", True), ("abc", False)]
        else:
            return False, f"unknown Python contract: {contract}"
        for value, expected in cases:
            actual = function(value)
            if actual != expected or type(actual) is not type(expected):
                return False, (
                    f"{function_name}({value!r}) returned {actual!r}, "
                    f"expected {expected!r}"
                )
    except Exception as exc:
        return False, f"function execution failed: {type(exc).__name__}: {exc}"
    return True, f"passed {len(cases)} restricted execution cases"


def _validate_output(prompt_spec: dict, text: str) -> tuple[bool, str]:
    """Validate one response with a deterministic, prompt-specific contract."""
    function_name = prompt_spec.get("python_function")
    if function_name:
        try:
            tree = ast.parse(_extract_python_source(text))
        except SyntaxError as exc:
            return False, f"invalid Python: {exc.msg}"

        return _run_restricted_python_contract(
            tree, function_name, prompt_spec.get("python_contract")
        )

    expected_number = prompt_spec.get("expected_number")
    if expected_number is not None:
        match = re.fullmatch(r"\s*(?:\*\*)?(-?\d+(?:\.\d+)?)(?:\*\*)?[.!]?\s*", text)
        if match and float(match.group(1)) == float(expected_number):
            return True, f"exact numeric answer {expected_number}"
        return False, f"expected only numeric answer {expected_number}"

    expected_sequence = prompt_spec.get("expected_number_sequence")
    if expected_sequence:
        numbers = [int(match) for match in re.findall(r"(?<!\w)-?\d+(?!\w)", text)]
        if numbers == expected_sequence:
            return True, f"exact numeric sequence {expected_sequence}"
        return False, f"expected only numeric sequence {expected_sequence}"

    expected_exact = prompt_spec.get("expected_exact", [])
    if expected_exact:
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        normalized = normalized.strip("`*_\"' .!?。！")
        if normalized in expected_exact:
            return True, f"exact normalized answer {normalized!r}"
        return False, f"expected one of {expected_exact}, got {normalized!r}"

    expected_terms = prompt_spec.get("expected_terms", [])
    if expected_terms:
        lower = text.lower()
        missing = [
            item
            for item in expected_terms
            if re.search(rf"(?<!\w){re.escape(item.lower())}(?!\w)", lower) is None
        ]
        if not missing:
            return True, f"contains terms {expected_terms}"
        return False, f"missing {missing}"

    min_length = prompt_spec.get("min_length")
    if min_length is not None:
        if len(text) >= min_length:
            return True, f"length={len(text)} >= {min_length}"
        return False, f"length={len(text)} < {min_length}"

    if text.strip():
        return True, "non-empty response"
    return False, "empty response"


def run_logprob_conformance(engine) -> dict:
    """Compare speculative decode logprobs with target-model prefill scores.

    This follows SGLang's speculative decoding conformance test: generate on the
    speculative path, then teacher-force the exact generated token ids through
    prefill scoring in the same engine. Free-form text equality is not a sound
    oracle after a near-tied BF16 token diverges, but target logprobs under the
    identical prefix are.
    """
    prompts = [
        "The capital of France is",
        "Explain quantum computing in simple terms:",
    ]
    logprob_delta_limit = 0.5
    choice_gap_limit = 0.05
    expected_tokens = len(prompts) * 32
    near_tie_limit = max(1, expected_tokens // 100)
    max_delta = 0.0
    large_gap_violations = 0
    near_ties = 0
    scored_tokens = 0
    errors = []

    for prompt in prompts:
        generated = engine.generate(
            prompt=_text_prompt(prompt),
            sampling_params={
                "temperature": 0,
                "max_new_tokens": 32,
                "ignore_eos": True,
            },
            return_logprob=True,
            top_logprobs_num=5,
            logprob_start_len=0,
        )
        meta = generated.get("meta_info", {})
        decode_logprobs = meta.get("output_token_logprobs") or []
        decode_top_logprobs = meta.get("output_top_logprobs") or []
        input_logprobs = meta.get("input_token_logprobs") or []
        input_token_ids = [entry[1] for entry in input_logprobs]
        output_token_ids = [entry[1] for entry in decode_logprobs]
        prompt_tokens = meta.get("prompt_tokens")
        if not decode_logprobs or not input_token_ids or prompt_tokens is None:
            errors.append(f"missing generation logprobs for {prompt!r}")
            continue
        if len(decode_top_logprobs) != len(decode_logprobs):
            errors.append(f"missing decode top-logprobs for {prompt!r}")
            continue
        if len(input_token_ids) != prompt_tokens:
            errors.append(
                f"generation prompt-token mismatch for {prompt!r}: "
                f"ids={len(input_token_ids)}, metadata={prompt_tokens}"
            )
            continue

        scored = engine.generate(
            input_ids=input_token_ids + output_token_ids,
            sampling_params={"temperature": 0, "max_new_tokens": 0},
            return_logprob=True,
            top_logprobs_num=5,
            logprob_start_len=0,
        )
        score_meta = scored.get("meta_info", {})
        score_prompt_tokens = score_meta.get("prompt_tokens")
        expected_score_tokens = len(input_token_ids) + len(output_token_ids)
        if score_prompt_tokens != expected_score_tokens:
            errors.append(
                f"score prompt-token mismatch for {prompt!r}: "
                f"metadata={score_prompt_tokens}, expected={expected_score_tokens}"
            )
            continue
        score_logprobs = (score_meta.get("input_token_logprobs") or [])[prompt_tokens:]
        score_top_logprobs = (score_meta.get("input_top_logprobs") or [])[
            prompt_tokens:
        ]
        if len(decode_logprobs) != len(score_logprobs):
            errors.append(
                f"logprob length mismatch for {prompt!r}: "
                f"decode={len(decode_logprobs)}, prefill={len(score_logprobs)}"
            )
            continue
        if len(score_top_logprobs) != len(score_logprobs):
            errors.append(f"missing prefill top-logprobs for {prompt!r}")
            continue

        for decode_entry, decode_top_entries, score_entry, top_entries in zip(
            decode_logprobs,
            decode_top_logprobs,
            score_logprobs,
            score_top_logprobs,
        ):
            decode_value = decode_entry[0]
            score_value = score_entry[0]
            chosen_id = decode_entry[1]
            if score_entry[1] != chosen_id:
                errors.append(
                    f"teacher-force token-id mismatch for {prompt!r}: "
                    f"decode={chosen_id}, prefill={score_entry[1]}"
                )
                continue
            if decode_value is None or score_value is None:
                errors.append(f"null token logprob for {prompt!r}")
                continue
            decode_value = float(decode_value)
            score_value = float(score_value)
            if not math.isfinite(decode_value) or not math.isfinite(score_value):
                errors.append(f"non-finite token logprob for {prompt!r}")
                continue
            delta = abs(decode_value - score_value)
            max_delta = max(max_delta, delta)
            scored_tokens += 1

            available_decode_top = [
                entry for entry in (decode_top_entries or []) if entry[0] is not None
            ]
            if not available_decode_top or any(
                not math.isfinite(float(entry[0])) for entry in available_decode_top
            ):
                errors.append(f"invalid decode top-logprobs for {prompt!r}")
                continue
            decode_chosen = [
                entry for entry in available_decode_top if entry[1] == chosen_id
            ]
            if not decode_chosen:
                errors.append(
                    f"chosen token missing from decode top-logprobs for {prompt!r}"
                )
                continue
            decode_top_value = max(float(entry[0]) for entry in available_decode_top)
            decode_chosen_value = max(float(entry[0]) for entry in decode_chosen)
            if decode_top_value - decode_chosen_value > 1e-6:
                errors.append(
                    f"temperature-0 decode did not choose top1 for {prompt!r}"
                )
                continue

            available_top = [
                entry for entry in (top_entries or []) if entry[0] is not None
            ]
            if not available_top:
                errors.append(f"empty top-logprobs for {prompt!r}")
                continue
            if any(not math.isfinite(float(entry[0])) for entry in available_top):
                errors.append(f"non-finite top-logprob for {prompt!r}")
                continue
            top_value, top_id = max(available_top, key=lambda entry: entry[0])[:2]
            if top_id != chosen_id:
                gap = float(top_value) - score_value
                if not math.isfinite(gap) or gap < 0:
                    errors.append(f"invalid target choice gap for {prompt!r}: {gap}")
                elif gap > choice_gap_limit:
                    large_gap_violations += 1
                else:
                    near_ties += 1

    return {
        "passed": not errors
        and scored_tokens == expected_tokens
        and max_delta < logprob_delta_limit
        and large_gap_violations == 0
        and near_ties <= near_tie_limit,
        "max_delta": max_delta,
        "logprob_delta_limit": logprob_delta_limit,
        "choice_gap_limit": choice_gap_limit,
        "large_gap_violations": large_gap_violations,
        "near_ties": near_ties,
        "near_tie_limit": near_tie_limit,
        "scored_tokens": scored_tokens,
        "errors": errors,
    }


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
        help="Disable CUDA graph capture (always forced on Ascend MTP)",
    )
    parser.add_argument(
        "--disable-piecewise-cuda-graph",
        action="store_true",
        help="Disable piecewise CUDA graph (always forced on Ascend MTP)",
    )
    parser.add_argument(
        "--disable-overlap-schedule",
        action="store_true",
        help="Use synchronous scheduling (always forced on Ascend MTP)",
    )
    args = parser.parse_args()
    _enforce_ascend_mtp_runtime_mode(args)

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        sys.exit(1)

    max_tokens = args.max_tokens
    disable_cg = args.disable_cuda_graph
    disable_pcg = args.disable_piecewise_cuda_graph
    mode_str = (
        "eager"
        if disable_cg
        else ("cuda_graph" if not disable_pcg else "cuda_graph(no piecewise)")
    )
    print("=" * 70)
    print("  Qwen3.6-27B MTP (Speculative Decoding) Validation")
    print("=" * 70)
    print(f"  Model: {MODEL_PATH}")
    print(f"  TP: {TP_SIZE} | max_tokens: {max_tokens} | mode: {mode_str}")
    print(f"  Requested overlap schedule: {not args.disable_overlap_schedule}")
    print("  MTP: algorithm=EAGLE, num_steps=3, topk=1, draft_tokens=4")
    print(f"  Prompts: {len(PROMPTS)} (factual/math/code/explanation/creative)")
    print()

    # ─── Phase 1: MTP inference ───────────────────────────────────────────────
    print("Phase 1: MTP-enabled inference")
    print("-" * 50)

    t0 = time.perf_counter()
    mtp_engine = _make_mtp_engine(
        disable_cuda_graph=disable_cg,
        disable_piecewise_cuda_graph=disable_pcg,
        disable_overlap_schedule=args.disable_overlap_schedule,
    )
    print(f"  Engine loaded in {time.perf_counter() - t0:.1f}s")

    # Platform compatibility defaults may resolve the requested mode (e.g.
    # MUSA MTP). Compare against a baseline using that same effective mode.
    effective_disable_overlap = mtp_engine.server_args.disable_overlap_schedule
    print(f"  Effective overlap schedule: {not effective_disable_overlap}")

    mtp_results = run_inference(mtp_engine, PROMPTS, max_tokens)

    print("\n  Target-token logprob conformance (decode vs prefill):")
    logprob_conformance = run_logprob_conformance(mtp_engine)
    print(
        "    "
        f"tokens={logprob_conformance['scored_tokens']}, "
        f"max_delta={logprob_conformance['max_delta']:.6f}, "
        f"near_ties={logprob_conformance['near_ties']}/"
        f"{logprob_conformance['near_tie_limit']}, "
        f"large_gap_violations={logprob_conformance['large_gap_violations']}"
    )
    for error in logprob_conformance["errors"]:
        print(f"    ERROR: {error}")

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
    _empty_device_cache()

    # ─── Phase 2: Baseline (optional) ────────────────────────────────────────
    baseline_results = None
    base_throughput = 0
    if not args.skip_baseline:
        print("\nPhase 2: Baseline (no MTP) inference")
        print("-" * 50)

        t0 = time.perf_counter()
        baseline_engine = _make_baseline_engine(
            disable_cuda_graph=disable_cg,
            disable_piecewise_cuda_graph=disable_pcg,
            disable_overlap_schedule=effective_disable_overlap,
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
        _empty_device_cache()

    # ─── Phase 3: Validation ──────────────────────────────────────────────────
    print("\nPhase 3: Validation")
    print("-" * 50)

    passed = 0
    failed = 0
    warnings = 0

    # 3a. Content correctness
    print("\n  [Content Correctness]")
    mtp_validity = []
    for p, (text, _, _) in zip(PROMPTS, mtp_results):
        valid, detail = _validate_output(p, text)
        mtp_validity.append(valid)
        if valid:
            print(f"    PASS [{p['category']:12}] {p['prompt'][:45]} ({detail})")
            passed += 1
        else:
            print(f"    FAIL [{p['category']:12}] {p['prompt'][:45]} ({detail})")
            print(f"         got: {text[:100]}")
            failed += 1

    # 3b. MTP vs baseline comparison. Full free-form text may diverge on BF16
    # hardware even with greedy decoding, so exact equality is diagnostic. The
    # strict gate requires both engines to satisfy the same semantic contract.
    if baseline_results:
        print("\n  [MTP vs Baseline Exact Match (greedy, temp=0; diagnostic)]")
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
            print("    WARN: <90% exact match; applying strict semantic contracts")
            warnings += 1

        print("\n  [MTP and Baseline Semantic Contract Agreement]")
        semantic_match_count = 0
        for p, mtp_valid, (base_text, _, _) in zip(
            PROMPTS, mtp_validity, baseline_results
        ):
            base_valid, base_detail = _validate_output(p, base_text)
            if mtp_valid and base_valid:
                semantic_match_count += 1
                print(f"    PASS [{p['category']:12}] both outputs satisfy contract")
            else:
                print(f"    FAIL [{p['category']:12}] semantic contract disagreement")
                print(f"         MTP valid: {mtp_valid}")
                print(f"         Base: {base_detail}")
        semantic_match_pct = semantic_match_count / len(PROMPTS) * 100
        print(
            "    Agreement: "
            f"{semantic_match_count}/{len(PROMPTS)} ({semantic_match_pct:.0f}%)"
        )
        if semantic_match_count == len(PROMPTS):
            print("    PASS: 100% semantic contract agreement")
            passed += 1
        else:
            print("    FAIL: <100% semantic contract agreement")
            failed += 1

    print("\n  [Target-Token Logprob Conformance]")
    if logprob_conformance["passed"]:
        print(
            "    PASS: speculative decode matches target prefill scoring "
            f"({logprob_conformance['scored_tokens']} tokens, "
            f"max delta {logprob_conformance['max_delta']:.6f})"
        )
        passed += 1
    else:
        print(
            "    FAIL: speculative decode/target prefill mismatch "
            f"(tokens={logprob_conformance['scored_tokens']}, "
            f"max delta={logprob_conformance['max_delta']:.6f}, "
            f"near ties={logprob_conformance['near_ties']}/"
            f"{logprob_conformance['near_tie_limit']}, "
            "large-gap violations="
            f"{logprob_conformance['large_gap_violations']})"
        )
        failed += 1

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
        print("    FAIL: stats not available")
        failed += 1

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
            print(
                "    WARN: MTP not faster; the sequential Ascend correctness "
                "fallback may trade throughput, so do not claim a speedup"
            )
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
