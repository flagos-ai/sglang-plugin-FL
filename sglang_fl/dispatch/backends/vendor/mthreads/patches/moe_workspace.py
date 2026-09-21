# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Reserve the fixed M16K scratch before SGLang measures available KV memory."""

import importlib
import inspect
import logging
from functools import wraps

from ..moe.dispatch import _enabled

logger = logging.getLogger(__name__)
_PATCH_MARKER = "_sglang_fl_musa_moe_pre_kv_workspace"


def _reserve(runner):
    from ..moe.fused_moe import maybe_preallocate_musa_m16k_moe_down_workspace

    return maybe_preallocate_musa_m16k_moe_down_workspace(
        runner.model_config,
        runner.device,
        tp_size=runner.tp_size,
        moe_ep_size=runner.moe_ep_size,
        is_draft_worker=runner.is_draft_worker,
    )


def _patch_runner(runner_cls):
    original = getattr(runner_cls, "init_memory_pool", None)
    if getattr(original, _PATCH_MARKER, False):
        return True
    if original is None:
        return False
    try:
        params = inspect.signature(original).parameters
    except (TypeError, ValueError):
        return False
    if tuple(params) != ("self", "pre_model_load_memory") or any(
        p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params.values()
    ):
        logger.warning("MUSA pre-KV workspace patch skipped: memory-pool ABI")
        return False

    @wraps(original)
    def init_memory_pool(self, pre_model_load_memory):
        if _enabled("SGLANG_MUSA_M16K_MOE_PREALLOCATE_DOWN_WORKSPACE"):
            _reserve(self)
        return original(self, pre_model_load_memory)

    setattr(init_memory_pool, _PATCH_MARKER, True)
    runner_cls.init_memory_pool = init_memory_pool
    return True


def apply_musa_moe_workspace_patch():
    if not _enabled("SGLANG_MUSA_M16K_MOE_PREALLOCATE_DOWN_WORKSPACE"):
        return False
    try:
        module = importlib.import_module("sglang.srt.model_executor.model_runner")
        runner_cls = module.ModelRunner
    except (ImportError, AttributeError) as exc:
        logger.warning("MUSA pre-KV workspace hook unavailable: %s", exc)
        return False
    applied = _patch_runner(runner_cls)
    if applied:
        logger.info("MUSA routed-MoE workspace hook installed before KV sizing")
    return applied
