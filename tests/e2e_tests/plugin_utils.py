# Copyright (c) 2026 BAAI. All rights reserved.

"""Shared helpers for sglang-plugin-FL end-to-end tests."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from tests.utils.model_config import ModelConfig, load_engine_overrides_from_env


def load_e2e_model_config() -> tuple[str, str, ModelConfig]:
    """Load and validate the model case selected by ``tests/run.py``."""
    import pytest

    model = os.environ.get("FL_TEST_MODEL", "")
    case = os.environ.get("FL_TEST_CASE", "")
    if not model or not case:
        pytest.skip(
            "FL_TEST_MODEL and FL_TEST_CASE must be set (injected by run.py)",
            allow_module_level=True,
        )

    config = ModelConfig.load(
        model,
        case,
        engine_overrides=load_engine_overrides_from_env(),
    )
    if not Path(config.model).exists():
        pytest.fail(f"Model not found: {config.model}", pytrace=False)
    return model, case, config


@lru_cache(maxsize=4)
def get_tokenizer(model_path: str) -> Any:
    """Load and cache a model tokenizer for E2E prompt rendering."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


@lru_cache(maxsize=4)
def get_processor(model_path: str) -> Any:
    """Load and cache a model processor for E2E multimodal prompts."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


def apply_chat_template(
    template_owner: Any,
    messages: list[dict[str, Any]],
) -> str:
    """Render chat messages, disabling thinking when the template supports it."""
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        return template_owner.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Non-Qwen templates may not accept the optional thinking argument.
        kwargs.pop("enable_thinking")
        return template_owner.apply_chat_template(messages, **kwargs)


def build_text_prompt(template_owner: Any, question: str) -> str:
    """Render one user question with the model's native chat template."""
    messages = [{"role": "user", "content": question}]
    return apply_chat_template(template_owner, messages)


def output_text(output: Any) -> str:
    """Extract generated text from an SGLang Engine output."""
    if isinstance(output, dict):
        return str(output.get("text", ""))
    return str(getattr(output, "text", ""))


def assert_expected(text: str, expected: Any, label: str) -> None:
    """Assert optional expected strings occur in a non-empty output."""
    assert text.strip(), f"Empty output for {label}"
    if not expected:
        return
    values = expected if isinstance(expected, list) else [expected]
    lower = text.lower()
    matched = any(str(item).lower() in lower for item in values)
    assert matched, f"Expected one of {values!r} in output for {label}, got: {text!r}"


def assert_sglang_fl_plugin_loaded_and_active() -> None:
    """Fail when the general sglang_fl plugin was not loaded and activated."""
    import sglang_fl

    assert sglang_fl.is_plugin_loaded(), (
        "sglang_fl is importable but its general plugin entry point was not invoked. "
        "Check the 'sglang.srt.plugins' entry-point registration."
    )
    assert sglang_fl.is_plugin_active(), (
        "sglang_fl is importable but its general plugin did not finish activation. "
        "Check the 'sglang.srt.plugins' entry-point registration."
    )
