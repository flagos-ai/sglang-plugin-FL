# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CoreX CUDA compatibility numbers are not NVIDIA SM capability levels."""

import importlib
import logging
import sys

logger = logging.getLogger(__name__)
_TARGET_MODULE = "sglang.srt.model_executor.model_runner_components.load_model_utils"
_MODEL_RUNNER_MODULE = "sglang.srt.model_executor.model_runner"
_FN_NAME = "maybe_downgrade_dtype_for_legacy_gpu"
_patched = False


def _downgrade_for_iluvatar(*, model_config, server_args=None) -> None:
    """Only fall back from BF16 when the vendor runtime reports no support."""
    import torch

    if model_config.dtype != torch.bfloat16:
        return

    # BI-V150 reports CUDA compatibility 7.1. Newer CoreX devices may support
    # BF16 despite that compatibility identity, so ask the vendor torch API.
    is_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if is_supported is None:
        raise RuntimeError(
            "CoreX torch cannot query BF16 support; select --dtype float16 "
            "or use a runtime with torch.cuda.is_bf16_supported"
        )
    try:
        supported = is_supported(including_emulation=False)
    except TypeError:
        # Older vendor builds expose the same API without this keyword.
        supported = is_supported()
    if supported:
        return

    from sglang.srt.runtime_context import get_context

    get_context().override("ModelRunner._sm80_dtype_fallback", dtype="float16")
    model_config.dtype = torch.float16
    logger.info("Iluvatar: vendor torch reports no BF16 support; using float16")


def patch_legacy_gpu_gate() -> None:
    global _patched
    if _patched:
        return

    module = importlib.import_module(_TARGET_MODULE)
    original = getattr(module, _FN_NAME)
    setattr(module, _FN_NAME, _downgrade_for_iluvatar)

    # ModelRunner copies the helper with a from-import. Cover both import orders.
    runner = sys.modules.get(_MODEL_RUNNER_MODULE)
    if runner is not None and getattr(runner, _FN_NAME, None) is original:
        setattr(runner, _FN_NAME, _downgrade_for_iluvatar)
    _patched = True
    logger.info("Iluvatar: replaced NVIDIA SM dtype gate with vendor BF16 query")
