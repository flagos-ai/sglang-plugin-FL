# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib
import logging
import sys

logger = logging.getLogger(__name__)

_TARGET = "sglang.srt.model_executor.model_runner_components.load_model_utils"
_MODEL_RUNNER = "sglang.srt.model_executor.model_runner"
_NAME = "maybe_downgrade_dtype_for_legacy_gpu"
_patched = False


def _downgrade_for_iluvatar(*, server_args, model_config) -> None:
    del server_args
    import torch

    if model_config.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        from sglang.srt.runtime_context import get_context

        get_context().override("ModelRunner._sm80_dtype_fallback", dtype="float16")
        model_config.dtype = torch.float16


def patch_legacy_gpu_gate() -> None:
    global _patched
    if _patched:
        return
    _patched = True

    module = importlib.import_module(_TARGET)
    original = getattr(module, _NAME)
    setattr(module, _NAME, _downgrade_for_iluvatar)
    model_runner = sys.modules.get(_MODEL_RUNNER)
    if model_runner is not None and getattr(model_runner, _NAME, None) is original:
        setattr(model_runner, _NAME, _downgrade_for_iluvatar)
    logger.info("iluvatar legacy-GPU gate patched")
