# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Match Qwen GDN convolution history to this runner's activation dtype."""

from dataclasses import replace
from functools import wraps
import importlib
import os


_MODEL_DTYPE = "_sglang_iluvatar_conv_model_dtype"
_GETTER_MARKER = "__sglang_iluvatar_conv_dtype__"


def _patch_qwen_cache_dtype(config_class):
    original = config_class.mamba2_cache_params
    if not isinstance(original, property) or original.fget is None:
        raise RuntimeError("Expected the SGLang Qwen mamba2_cache_params property")
    if getattr(original.fget, _GETTER_MARKER, False):
        return

    @wraps(original.fget)
    def cache_params(config):
        params = original.fget(config)
        # Explicit user state precision retains upstream precedence. A marker is
        # attached only to the current Iluvatar runner's HF text config, never
        # to the process environment or another model instance.
        if "SGLANG_MAMBA_CONV_DTYPE" in os.environ:
            return params
        import torch

        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(getattr(config, _MODEL_DTYPE, None))
        if dtype is None or params.dtype.conv == dtype:
            return params
        # Both records are frozen dataclasses. Preserve state precision and all
        # shape/layer metadata; only convolution history stores activations.
        return replace(params, dtype=replace(params.dtype, conv=dtype))

    setattr(cache_params, _GETTER_MARKER, True)
    config_class.mamba2_cache_params = property(
        cache_params, original.fset, original.fdel, original.__doc__
    )


def record_model_conv_dtype(model_config):
    """Record the final effective model dtype after the Iluvatar BF16 gate."""
    config = getattr(model_config, "hf_text_config", None)
    if config is None:
        return
    module = importlib.import_module("sglang.srt.configs.qwen3_next")
    config_class = module.Qwen3NextConfig
    if not isinstance(config, config_class):
        return
    import torch

    name = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }.get(model_config.dtype)
    if name is None:
        return
    _patch_qwen_cache_dtype(config_class)
    # A plain string remains serializable in HF config diagnostics.
    setattr(config, _MODEL_DTYPE, name)
