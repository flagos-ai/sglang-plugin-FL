# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Use SGLang's torch clamp implementation instead of NVIDIA nvcc JIT."""

import importlib
import logging

logger = logging.getLogger(__name__)
_applied = False


def patch_clamp_position() -> None:
    global _applied
    if _applied:
        return

    module = importlib.import_module("sglang.srt.model_executor.forward_batch_info")
    module.clamp_position = module._clamp_position_native
    _applied = True
    logger.info("Iluvatar: clamp_position uses SGLang's torch implementation")
