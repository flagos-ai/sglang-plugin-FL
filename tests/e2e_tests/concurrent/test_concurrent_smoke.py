# Copyright (c) 2026 BAAI. All rights reserved.

"""Concurrent inference smoke test driven by model YAML configs.

This test mirrors the concurrent examples in ``examples/`` and runs one or more
N-way async generation modes against a single SGLang Engine instance.

The default mode is intentionally small for CI:
- text model configs run text concurrency
- image model configs run VL concurrency

Set ``FL_CONCURRENT_MODES=text,vl,mixed`` or ``FL_CONCURRENT_MODES=all`` to run
more modes without changing the YAML files.
"""

import asyncio
import os
import statistics
import time
from pathlib import Path
from typing import Any

import pytest

from sglang import Engine
from tests.utils.model_config import ModelConfig

_REPO_ROOT = Path(__file__).resolve().parents[3]
_IMAGE_DIR = Path(os.environ.get("IMAGE_DIR", _REPO_ROOT / "examples" / "test_images"))

_MODEL = os.environ.get("FL_TEST_MODEL", "")
_CASE = os.environ.get("FL_TEST_CASE", "")

if not _MODEL or not _CASE:
    pytest.skip(
        "FL_TEST_MODEL and FL_TEST_CASE must be set (injected by run.py)",
        allow_module_level=True,
    )

_CFG = ModelConfig.load(_MODEL, _CASE)

if not os.path.exists(_CFG.model):
    pytest.fail(f"Model not found: {_CFG.model}", pytrace=False)

_DEFAULT_TEXT_CASES = [
    {
        "text": "How many states are there in the United States?",
        "expected": ["50"],
    },
    {"text": "The capital of France is", "expected": ["paris"]},
    {
        "text": "What is the largest planet in the solar system?",
        "expected": ["jupiter"],
    },
    {
        "text": "Who wrote the play Romeo and Juliet?",
        "expected": ["shakespeare"],
    },
    {"text": "What is 17 multiplied by 13?", "expected": ["221"]},
    {
        "text": "Name the three primary colors of light.",
        "expected": ["red", "green", "blue"],
    },
    {"text": "What year did World War II end?", "expected": ["1945"]},
    {
        "text": "Explain the concept of gravity in one sentence.",
        "expected": ["mass", "force", "attract"],
    },
]

_DEFAULT_VL_CASES = [
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


def _normalize_text_cases(raw_cases: list[Any]) -> list[dict[str, Any]]:
    if not raw_cases:
        return _DEFAULT_TEXT_CASES

    cases = []
    for item in raw_cases:
        if isinstance(item, str):
            cases.append({"text": item, "expected": []})
            continue
        expected = item.get("expected", [])
        if isinstance(expected, str):
            expected = [expected]
        cases.append({"text": item["text"], "expected": expected})
    return cases


_TEXT_CASES = _normalize_text_cases(_CFG.concurrent.text_prompts)
TEXT_PROMPTS = [case["text"] for case in _TEXT_CASES]
TEXT_EXPECTED = {
    case["text"]: case["expected"] for case in _TEXT_CASES if case.get("expected")
}
VL_CASES = _CFG.concurrent.vl_cases or _DEFAULT_VL_CASES

ALL_MODES = ["text", "vl", "mixed"]
CONCURRENT_N = int(
    os.environ.get(
        "FL_CONCURRENT_N",
        os.environ.get("CONCURRENT_N", str(_CFG.concurrent.concurrent_n)),
    )
)
_TEXT_SAMPLING = {"max_new_tokens": 64, "temperature": 0}
_TEXT_SAMPLING.update(_CFG.concurrent.text_sampling)
_TEXT_SAMPLING["max_new_tokens"] = int(
    os.environ.get("FL_CONCURRENT_MAX_TOKENS", _TEXT_SAMPLING["max_new_tokens"])
)

_VL_SAMPLING = {"max_new_tokens": 64, "temperature": 0}
_VL_SAMPLING.update(_CFG.concurrent.vl_sampling)
_VL_SAMPLING["max_new_tokens"] = int(
    os.environ.get("FL_CONCURRENT_VL_MAX_TOKENS", _VL_SAMPLING["max_new_tokens"])
)

_tokenizer = None
_processor = None


def _selected_modes() -> list[str]:
    raw = os.environ.get("FL_CONCURRENT_MODES", "").strip()
    if raw:
        modes = ALL_MODES if raw == "all" else [m.strip() for m in raw.split(",")]
        invalid = [m for m in modes if m not in ALL_MODES]
        assert not invalid, f"Unsupported FL_CONCURRENT_MODES values: {invalid}"
        return [m for m in modes if m]
    if _CFG.concurrent.modes:
        return _CFG.concurrent.modes
    return ["vl"] if _CFG.generate.modality == "image" else ["text"]


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(_CFG.model, trust_remote_code=True)
    return _tokenizer


def _get_processor():
    global _processor
    if _processor is None:
        from transformers import AutoProcessor

        _processor = AutoProcessor.from_pretrained(_CFG.model, trust_remote_code=True)
    return _processor


def _apply_chat_template(template_owner, messages: list[dict[str, Any]]) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        return template_owner.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return template_owner.apply_chat_template(messages, **kwargs)


def _text_prompt(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    return _apply_chat_template(_get_tokenizer(), messages)


def _image_uri(name: str) -> str:
    path = Path(name)
    if not path.is_absolute():
        repo_relative = (_REPO_ROOT / path).resolve()
        image_relative = (_IMAGE_DIR / path).resolve()
        path = repo_relative if repo_relative.is_file() else image_relative
    assert path.is_file(), f"Missing test image: {path}"
    return f"file://{path.resolve()}"


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
    return _apply_chat_template(_get_processor(), messages)


def _output_text(output: Any) -> str:
    if isinstance(output, dict):
        return str(output.get("text", ""))
    return str(getattr(output, "text", ""))


def _completion_tokens(output: Any) -> int:
    if not isinstance(output, dict):
        return 0
    return int(output.get("meta_info", {}).get("completion_tokens", 0) or 0)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * p
    floor = int(index)
    ceil = min(floor + 1, len(sorted_values) - 1)
    return sorted_values[floor] + (sorted_values[ceil] - sorted_values[floor]) * (index - floor)


def _report(label: str, n_req: int, elapsed: float, latencies: list[float], total_tokens: int) -> None:
    mean = statistics.fmean(latencies) if latencies else 0.0
    p50 = _percentile(latencies, 0.50)
    p99 = _percentile(latencies, 0.99)
    print(
        f"\n[{label}] {n_req} req | wall {elapsed:.2f}s | "
        f"{n_req / elapsed:.2f} req/s | {total_tokens / elapsed:.1f} tok/s | "
        f"latency mean {mean:.2f}s P50 {p50:.2f}s P99 {p99:.2f}s"
    )


def _assert_expected(text: str, expected: list[str], label: str) -> None:
    assert text.strip(), f"Empty output for {label}"
    lower = text.lower()
    matched = any(item.lower() in lower for item in expected)
    assert matched, f"Expected one of {expected!r} in output for {label}, got: {text!r}"


def _validate_text(pairs: list[tuple[str, str]]) -> None:
    for prompt, text in pairs:
        assert text.strip(), f"Empty output for {prompt}"
        expected = TEXT_EXPECTED.get(prompt, [])
        if expected:
            _assert_expected(text, expected, prompt)
    print("  text validation passed.")


def _validate_vl(pairs: list[tuple[dict[str, Any], str]]) -> None:
    for case, text in pairs:
        _assert_expected(text, case["expected"], case["image"])
    print("  vl validation passed.")


def _check_images() -> None:
    missing = []
    for case in VL_CASES:
        try:
            _image_uri(case["image"])
        except AssertionError:
            missing.append(case["image"])
    assert not missing, "Missing test images: " + ", ".join(missing)


def _engine_kwargs() -> dict[str, Any]:
    kwargs = _CFG.engine_kwargs()
    kwargs.setdefault("disable_cuda_graph", True)
    kwargs.setdefault("disable_piecewise_cuda_graph", True)
    return kwargs


def _run_text_concurrent(engine: Engine) -> list[tuple[str, str]]:
    base = [(prompt, _text_prompt(prompt)) for prompt in TEXT_PROMPTS]
    items = [base[i % len(base)] for i in range(CONCURRENT_N)]

    async def one(prompt_text: str):
        start = time.perf_counter()
        output = await engine.async_generate(prompt=prompt_text, sampling_params=_TEXT_SAMPLING)
        return time.perf_counter() - start, output

    async def run():
        start = time.perf_counter()
        results = await asyncio.gather(*(one(prompt_text) for _, prompt_text in items))
        return time.perf_counter() - start, results

    elapsed, results = engine.loop.run_until_complete(run())
    latencies = [latency for latency, _ in results]
    total_tokens = sum(_completion_tokens(output) for _, output in results)

    for (label, _), (_, output) in zip(items[: min(len(items), len(base))], results):
        print(f"  {label!r}\n    -> {_output_text(output)!r}")

    _report("text-concurrent", CONCURRENT_N, elapsed, latencies, total_tokens)
    return [(label, _output_text(output)) for (label, _), (_, output) in zip(items, results)]


def _run_vl_concurrent(engine: Engine) -> list[tuple[dict[str, Any], str]]:
    _check_images()
    base = []
    for case in VL_CASES:
        uri = _image_uri(case["image"])
        base.append((case, _vl_prompt(case["question"], uri), uri))
    items = [base[i % len(base)] for i in range(CONCURRENT_N)]

    async def one(prompt_text: str, image_uri: str):
        start = time.perf_counter()
        output = await engine.async_generate(
            prompt=prompt_text,
            image_data=[image_uri],
            sampling_params=_VL_SAMPLING,
        )
        return time.perf_counter() - start, output

    async def run():
        start = time.perf_counter()
        results = await asyncio.gather(*(one(prompt_text, uri) for _, prompt_text, uri in items))
        return time.perf_counter() - start, results

    elapsed, results = engine.loop.run_until_complete(run())
    latencies = [latency for latency, _ in results]
    total_tokens = sum(_completion_tokens(output) for _, output in results)

    for (case, _, _), (_, output) in zip(items[: min(len(items), len(base))], results):
        print(f"  [{case['image']}] {case['question']}\n    -> {_output_text(output)!r}")

    _report("vl-concurrent", CONCURRENT_N, elapsed, latencies, total_tokens)
    return [(case, _output_text(output)) for (case, _, _), (_, output) in zip(items, results)]


def _run_mixed_concurrent(engine: Engine) -> tuple[list[tuple[str, str]], list[tuple[dict[str, Any], str]]]:
    _check_images()
    n_text = CONCURRENT_N // 2
    n_vl = CONCURRENT_N - n_text

    text_base = [(prompt, _text_prompt(prompt)) for prompt in TEXT_PROMPTS]
    text_items = [text_base[i % len(text_base)] for i in range(n_text)]

    vl_base = []
    for case in VL_CASES:
        uri = _image_uri(case["image"])
        vl_base.append((case, _vl_prompt(case["question"], uri), uri))
    vl_items = [vl_base[i % len(vl_base)] for i in range(n_vl)]

    async def text_one(prompt_text: str):
        start = time.perf_counter()
        output = await engine.async_generate(prompt=prompt_text, sampling_params=_TEXT_SAMPLING)
        return time.perf_counter() - start, output

    async def vl_one(prompt_text: str, image_uri: str):
        start = time.perf_counter()
        output = await engine.async_generate(
            prompt=prompt_text,
            image_data=[image_uri],
            sampling_params=_VL_SAMPLING,
        )
        return time.perf_counter() - start, output

    async def run():
        start = time.perf_counter()
        outputs = await asyncio.gather(
            *(text_one(prompt_text) for _, prompt_text in text_items),
            *(vl_one(prompt_text, uri) for _, prompt_text, uri in vl_items),
        )
        return time.perf_counter() - start, outputs[:n_text], outputs[n_text:]

    elapsed, text_results, vl_results = engine.loop.run_until_complete(run())
    latencies = [latency for latency, _ in text_results] + [latency for latency, _ in vl_results]
    total_tokens = sum(_completion_tokens(output) for _, output in [*text_results, *vl_results])

    _report("mixed-concurrent", CONCURRENT_N, elapsed, latencies, total_tokens)
    text_pairs = [(label, _output_text(output)) for (label, _), (_, output) in zip(text_items, text_results)]
    vl_pairs = [(case, _output_text(output)) for (case, _, _), (_, output) in zip(vl_items, vl_results)]
    return text_pairs, vl_pairs


@pytest.mark.e2e
def test_concurrent() -> None:
    """Smoke test: run concurrent async generation and validate outputs."""
    modes = _selected_modes()
    print(f"\n[{_MODEL}/{_CASE}] concurrent modes: {modes}")
    print(f"[{_MODEL}/{_CASE}] model: {_CFG.model}")
    print(f"[{_MODEL}/{_CASE}] concurrent_n: {CONCURRENT_N}")

    engine = Engine(**_engine_kwargs())
    try:
        for index, mode in enumerate(modes):
            if index > 0 and hasattr(engine, "flush_cache"):
                engine.flush_cache()
            print(f"\n=== concurrent mode: {mode} ===")
            if mode == "text":
                _validate_text(_run_text_concurrent(engine))
            elif mode == "vl":
                _validate_vl(_run_vl_concurrent(engine))
            elif mode == "mixed":
                text_pairs, vl_pairs = _run_mixed_concurrent(engine)
                _validate_text(text_pairs)
                _validate_vl(vl_pairs)
            else:
                raise AssertionError(f"Unsupported concurrent mode: {mode}")
    finally:
        engine.shutdown()






