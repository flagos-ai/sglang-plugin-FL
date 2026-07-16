# Copyright (c) 2026 BAAI. All rights reserved.

"""
Unified inference smoke test auto-generated from model YAML configs.

This test file is driven by two environment variables set by ``tests/run.py``:

- ``FL_TEST_MODEL``: Model family (e.g. ``qwen3``, ``minicpm``)
- ``FL_TEST_CASE``:  Case name within the family (e.g. ``06b_tp1``, ``o45_tp2``)

It loads ``tests/models/<model>/<case>.yaml``, constructs the SGLang Engine,
runs generation for each prompt (with optional parametrize combos), and
asserts that outputs are non-empty.

Supports both text-only and multimodal (audio/image/video) models via the
``generate.modality`` field in the YAML config.
"""

from pathlib import Path
import os

import pytest

from tests.utils.model_config import ModelConfig
from sglang import Engine

_REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Load config from environment (injected by run.py)
# ---------------------------------------------------------------------------

_MODEL = os.environ.get("FL_TEST_MODEL", "")
_CASE = os.environ.get("FL_TEST_CASE", "")

if not _MODEL or not _CASE:
    pytest.skip(
        "FL_TEST_MODEL and FL_TEST_CASE must be set (injected by run.py)",
        allow_module_level=True,
    )

_CFG = ModelConfig.load(_MODEL, _CASE)

if not os.path.exists(_CFG.model):
    pytest.fail(
        f"Model not found: {_CFG.model}",
        pytrace=False,
    )


# ---------------------------------------------------------------------------
# Multimodal helpers
# ---------------------------------------------------------------------------


def _resolve_asset_uri(asset: str) -> str:
    """Resolve a local asset path or URL into a SGLang image/video/audio input."""
    if asset.startswith(("http://", "https://", "file://", "data:")):
        return asset

    path = Path(asset)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    assert path.is_file(), f"Multimodal asset not found: {path}"
    return f"file://{path.resolve()}"


def _load_assets(modality: str, asset_names: list[str], count: int) -> dict:
    """Load multimodal assets for SGLang Engine input.

    Args:
        modality: One of ``audio``, ``image``, ``video``.
        asset_names: Pool of asset paths or URLs from YAML config.
        count: Number of assets to include (0 returns empty dict).

    Returns:
        Dict suitable for SGLang ``generate`` keyword arguments.
    """
    if count <= 0:
        return {}

    selected = [_resolve_asset_uri(asset) for asset in asset_names[:count]]
    key_map = {
        "audio": "audio_data",
        "image": "image_data",
        "video": "video_data",
    }
    if modality not in key_map:
        raise ValueError(f"Unsupported modality: {modality}")
    return {key_map[modality]: selected}


def _build_multimodal_prompt(
    tokenizer,
    question: str,
    modality: str,
    asset_count: int,
) -> str:
    """Build a chat prompt with multimodal placeholders.

    Uses the tokenizer's ``apply_chat_template`` to format the prompt.
    The placeholder format is determined by modality.
    """
    placeholder_map = {
        "audio": "(<audio>./</audio>)",
        "image": "(<image>./</image>)",
        "video": "(<video>./</video>)",
    }
    placeholder = placeholder_map.get(modality, "")
    content = (
        f"{placeholder * asset_count}\n{question}" if asset_count > 0 else question
    )

    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _build_image_prompt(processor, question: str, image_uri: str) -> str:
    """Build an image prompt using the Qwen-style processor chat template."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": question},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# ---------------------------------------------------------------------------
# Test function
# ---------------------------------------------------------------------------


def _get_max_asset_count() -> int:
    """Return the maximum asset_count across all multimodal prompts."""
    gen = _CFG.generate
    if gen.modality == "text":
        return 0
    return max(
        (p.get("asset_count", 0) for p in gen.prompts if isinstance(p, dict)),
        default=0,
    )


def _output_text(output) -> str:
    """Extract generated text from a SGLang Engine output object."""
    if isinstance(output, dict):
        return str(output.get("text", ""))
    return str(getattr(output, "text", ""))


def _assert_expected(text: str, expected, label: str) -> None:
    """Assert an optional string/list expected value appears in the output."""
    if not expected:
        return
    values = expected if isinstance(expected, list) else [expected]
    lower = text.lower()
    matched = any(str(item).lower() in lower for item in values)
    assert matched, f"Expected one of {values!r} in output for {label}, got: {text!r}"


def _run_text_test(llm: Engine, sampling_params: dict) -> None:
    """Run text-only generation test.

    Prompts can be plain strings or dicts with ``text`` and optional
    ``expected`` (substring that must appear in the output).
    """
    gen = _CFG.generate
    raw_prompts = gen.prompts
    assert len(raw_prompts) > 0, "No prompts defined in YAML config"

    # Normalize: str -> {"text": str}, dict stays as-is
    prompt_cfgs = []
    for p in raw_prompts:
        if isinstance(p, str):
            prompt_cfgs.append({"text": p})
        else:
            prompt_cfgs.append(p)

    prompt_texts = [p["text"] for p in prompt_cfgs]
    outputs = llm.generate(prompt_texts, sampling_params)
    assert len(outputs) == len(prompt_cfgs), (
        f"Expected {len(prompt_cfgs)} outputs, got {len(outputs)}"
    )

    for i, output in enumerate(outputs):
        text = _output_text(output)
        prompt = prompt_cfgs[i]["text"]
        expected = prompt_cfgs[i].get("expected")

        assert len(text) > 0, f"Empty output for prompt[{i}]: {prompt}"
        print(f"  prompt[{i}]: {prompt!r}")
        print(f"  output[{i}]: {text!r}")
        _assert_expected(text, expected, f"prompt[{i}]")


def _run_multimodal_test(llm: Engine, sampling_params: dict) -> None:
    """Run multimodal generation test."""
    gen = _CFG.generate

    if gen.modality == "image":
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(_CFG.model, trust_remote_code=True)
        for i, prompt_cfg in enumerate(gen.prompts):
            assert isinstance(prompt_cfg, dict), (
                f"Image prompts must be dicts, got: {type(prompt_cfg)}"
            )
            question = prompt_cfg["question"]
            image_uri = _resolve_asset_uri(prompt_cfg["image"])
            prompt_text = _build_image_prompt(processor, question, image_uri)
            output = llm.generate(
                prompt=prompt_text,
                image_data=[image_uri],
                sampling_params=sampling_params,
            )

            text = _output_text(output)
            assert len(text) > 0, f"Empty output for image prompt[{i}]"
            print(f"  [image] {prompt_cfg['image']}: {question}")
            print(f"  Output: {text!r}")
            _assert_expected(text, prompt_cfg.get("expected"), f"image prompt[{i}]")
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        _CFG.model,
        trust_remote_code=True,
    )

    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    stop_ids = []
    for t in stop_tokens:
        token_id = tokenizer.convert_tokens_to_ids(t)
        if isinstance(token_id, int) and token_id != tokenizer.unk_token_id:
            stop_ids.append(token_id)

    if stop_ids:
        sampling_params = _CFG.sampling_kwargs(stop_token_ids=stop_ids)

    for i, prompt_cfg in enumerate(gen.prompts):
        assert isinstance(prompt_cfg, dict), (
            f"Multimodal prompts must be dicts, got: {type(prompt_cfg)}"
        )
        question = prompt_cfg["question"]
        asset_count = prompt_cfg.get("asset_count", 0)

        prompt_text = _build_multimodal_prompt(
            tokenizer,
            question,
            gen.modality,
            asset_count,
        )
        mm_kwargs = _load_assets(gen.modality, gen.assets, asset_count)

        output = llm.generate(
            prompt=prompt_text,
            sampling_params=sampling_params,
            **mm_kwargs,
        )

        text = _output_text(output)
        assert isinstance(text, str), f"Output is not str for prompt[{i}]"
        assert len(text) > 0, f"Empty output for prompt[{i}]"

        print(f"  [{gen.modality} count={asset_count}] Q: {question}")
        print(f"  Output: {text!r}")
        _assert_expected(text, prompt_cfg.get("expected"), f"prompt[{i}]")


def _run_configured_vl_test(llm: Engine) -> None:
    """Run optional image cases from generate.vl."""
    vl = _CFG.generate.vl
    cases = vl.get("cases", [])
    if not cases:
        return

    print("\n=== generate.vl inference ===")

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(_CFG.model, trust_remote_code=True)
    sampling_params = dict(_CFG.generate.sampling)
    sampling_params.update(vl.get("sampling", {}))

    for i, case in enumerate(cases):
        image_uri = _resolve_asset_uri(case["image"])
        prompt_text = _build_image_prompt(processor, case["question"], image_uri)
        output = llm.generate(
            prompt=prompt_text,
            image_data=[image_uri],
            sampling_params=sampling_params,
        )
        text = _output_text(output)

        assert text.strip(), f"Empty output for generate.vl case[{i}]"
        _assert_expected(text, case.get("expected"), f"generate.vl case[{i}]")
        print(f"  [generate.vl/{i}] {case['image']}: {case['question']}")
        print(f"  output[{i}]: {text!r}")


# ---------------------------------------------------------------------------
# Parametrized test entry point
# ---------------------------------------------------------------------------


_COMBOS = _CFG.generate.get_parametrize_combos()
_COMBO_IDS = [
    "-".join(f"{k}={v}" for k, v in combo.items()) or "default" for combo in _COMBOS
]


@pytest.mark.e2e
@pytest.mark.parametrize("combo", _COMBOS, ids=_COMBO_IDS)
def test_inference(combo: dict) -> None:
    """Smoke test: load model, generate, assert non-empty output.

    Each parametrize combo overrides engine params from the YAML config.
    """
    gen = _CFG.generate

    # Build Engine kwargs with parametrize overrides
    llm_kwargs = _CFG.engine_kwargs(**combo)

    # For multimodal: inject limit_mm_per_prompt
    max_assets = _get_max_asset_count()
    if gen.modality != "text" and max_assets > 0:
        llm_kwargs.setdefault(
            "limit_mm_per_prompt",
            {
                "image": 0,
                "video": 0,
                "audio": 0,
                gen.modality: max_assets,
            },
        )

    combo_desc = ", ".join(f"{k}={v}" for k, v in combo.items()) or "default"
    print(f"\n[{_MODEL}/{_CASE}] combo: {combo_desc}")
    print(f"[{_MODEL}/{_CASE}] model: {_CFG.model}")

    llm = Engine(**llm_kwargs)
    try:
        sampling_params = _CFG.sampling_kwargs()

        if gen.modality == "text":
            _run_text_test(llm, sampling_params)
        else:
            _run_multimodal_test(llm, sampling_params)

        _run_configured_vl_test(llm)
    finally:
        llm.shutdown()
