# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monkey patches to adapt sglang core for txda (TsingMicro) platform.

These patches replace direct modifications to the sglang source tree with
runtime monkey-patching, keeping all txda-specific logic inside the plugin.

Patches applied:
  device_support  – torchvision fake-register tolerance, safe CUDA
                    properties, get_device() txda awareness, inject
                    is_txda() utility, SUPPORTED_DEVICES txda extension
  dist_init       – GroupCoordinator device detection, distributed backend
                    map, init_distributed_environment backend override,
                    all_gather_into_tensor txda routing
  per_rank_log    – Redirect scheduler subprocess stdout/stderr to
                    per-rank log files (SGLANG_FL_LOG_DIR)
  fused_moe       – Patch moe_align_block_size import, disable vllm_ops,
                    bypass register_custom_op for inplace_fused_experts,
                    route UnquantizedFusedMoEMethod.apply through dispatch
  platform_stubs  – Inject stub modules for CUDA-only deps on non-CUDA platforms
  model_runner    – Per-iteration forward() timing (SGLANG_FL_TIMER_ENABLE)

Usage (called automatically by load_plugin()):

    from sglang_fl.dispatch.backends.vendor.txda.patches import apply_all_txda_patches
    apply_all_txda_patches()
"""

import torch

from sglang_fl.dispatch.backends.vendor.txda.patches._logger import patch_logger
from sglang_fl.dispatch.backends.vendor.txda.patches._utils import is_txda as is_txda_available

_log = patch_logger("patches")


def apply_all_txda_patches() -> None:
    """Apply all monkey patches required for txda platform inference.

    Idempotent: safe to call multiple times.  Non-txda platforms are no-ops
    for per-patch submodules; per_rank_log and device_support are applied
    unconditionally.
    """
    from sglang_fl.dispatch.backends.vendor.txda.patches import per_rank_log, device_support, platform_stubs, model_runner

    per_rank_log.patch()
    device_support.patch()   # universal + txda-gated internally
    platform_stubs.patch()   # dummy-module injection for non-CUDA platforms
    model_runner.patch()     # gated by SGLANG_FL_TIMER_ENABLE internally

    if not is_txda_available():
        _log.skipped("txda not available — txda-specific patches skipped")
        return

    from sglang_fl.dispatch.backends.vendor.txda.patches import dist_init, fused_moe

    dist_init.patch()
    fused_moe.patch()  # includes former unquant patch

    _log.applied("all txda monkey patches applied")
