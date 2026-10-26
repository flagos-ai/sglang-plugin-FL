"""Run MetaX model-parallel all-reduces in FP32 for low-precision tensors."""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_patched = False


def patch_fp32_tp_allreduce() -> None:
    """Wrap all-reduce when SGLANG_FL_METAX_FP32_ALLREDUCE is enabled."""
    global _patched
    if _patched:
        return
    if os.environ.get("SGLANG_FL_METAX_FP32_ALLREDUCE", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def all_reduce_fp32(original_fn, group, input_: torch.Tensor) -> torch.Tensor:
        if group.world_size == 1 or input_.dtype not in (
            torch.bfloat16,
            torch.float16,
        ):
            return original_fn(group, input_)

        return original_fn(group, input_.float()).to(input_.dtype)

    HookRegistry.register(
        "sglang.srt.distributed.parallel_state.GroupCoordinator.all_reduce",
        all_reduce_fp32,
        HookType.AROUND,
    )
    _patched = True
    logger.info("registered MetaX FP32 hook for GroupCoordinator.all_reduce")
