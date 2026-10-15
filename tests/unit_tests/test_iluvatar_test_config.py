# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.utils.model_config import ModelConfig
from tests.utils.platform_config import PlatformConfig


@pytest.mark.parametrize(
    "case,expected",
    [
        ("27b_tp4_nograph", "/models/Qwen3.6-27B"),
        ("35b_a3b_tp4_nograph", "/models/Qwen3.6-35B-A3B"),
    ],
)
def test_platform_paths_match_engine_and_tokenizer(case, expected):
    platform = PlatformConfig.load("iluvatar", "bi_v150")
    config = ModelConfig.load("qwen3_6", case)
    overrides = platform.get_engine_overrides("qwen3_6", case)
    original = dict(overrides)
    config.apply_engine_overrides(overrides)
    assert config.model == expected
    assert config.engine_kwargs()["model_path"] == expected
    assert "model" not in config.engine_kwargs()
    assert config.engine_kwargs()["dtype"] == "float16"
    assert overrides == original


def test_other_platform_engine_overrides_preserve_model():
    config = ModelConfig(model="/existing/model")
    config.apply_engine_overrides({"page_size": 1})
    assert config.model == "/existing/model"
    assert config.engine_kwargs() == {"model_path": "/existing/model", "page_size": 1}
