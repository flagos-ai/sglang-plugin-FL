# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Skip cold softmax-TopK autotuning for measured MTT S5000 shapes.

The MUSA kernel carries fifteen launch configurations.  Autotuning them also
flushes a 256 MiB cache buffer between candidates, making the first large
prefill wave several seconds slower.  On MP31, ``warps=1, stages=1`` is the
validated choice for Qwen3.6-35B-A3B's ``E=256, K=8`` graph and prefill shapes.

Only those exact shapes are pinned.  Other models and shapes keep SGLang's
normal autotuner.  Set ``SGLANG_MUSA_TOPK_SCHEDULE=off`` to disable the patch.
"""

from __future__ import annotations

import logging
import os
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)

_ENV_NAME = "SGLANG_MUSA_TOPK_SCHEDULE"
_PATCH_MARKER = "_sglang_fl_musa_topk_schedule"
_TARGET_GRAPH_TOKEN_COUNTS = frozenset(
    {
        # CUDA-graph capture batches.
        1,
        2,
        4,
        8,
        12,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
    }
)
_TARGET_PREFILL_TOKEN_RANGE = range(1024, 16385)


def _enabled() -> bool:
    return os.environ.get(_ENV_NAME, "auto").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "disable",
        "disabled",
    }


def _device_name() -> str:
    try:
        import torch
    except ImportError:
        return ""

    try:
        if hasattr(torch, "musa") and torch.musa.is_available():
            return str(torch.musa.get_device_name())
    except RuntimeError:
        logger.debug("Unable to query MUSA device name", exc_info=True)
    return ""


def _is_target_shape(
    topk_weights: Any,
    gating_output: Any,
    moe_softcapping: float,
    correction_bias: Any,
) -> bool:
    try:
        num_tokens = int(gating_output.shape[0])
        return (
            gating_output.ndim == 2
            and (
                num_tokens in _TARGET_GRAPH_TOKEN_COUNTS
                or num_tokens in _TARGET_PREFILL_TOKEN_RANGE
            )
            and int(gating_output.shape[1]) == 256
            and int(topk_weights.shape[-1]) == 8
            and not moe_softcapping
            and correction_bias is None
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        return False


def _make_topk_wrapper(
    original: Callable[..., Any], kernel: Any, selected_config: Any
) -> Callable[..., Any]:
    @wraps(original)
    def wrapped(
        topk_weights: Any,
        topk_ids: Any,
        gating_output: Any,
        renormalize: bool = False,
        moe_softcapping: float = 0,
        correction_bias: Any = None,
    ) -> Any:
        if not _is_target_shape(
            topk_weights, gating_output, moe_softcapping, correction_bias
        ):
            return original(
                topk_weights,
                topk_ids,
                gating_output,
                renormalize,
                moe_softcapping,
                correction_bias,
            )

        original_configs = kernel.configs
        kernel.configs = [selected_config]
        try:
            return original(
                topk_weights,
                topk_ids,
                gating_output,
                renormalize,
                moe_softcapping,
                correction_bias,
            )
        finally:
            kernel.configs = original_configs

    setattr(wrapped, _PATCH_MARKER, True)
    return wrapped


def apply_musa_topk_schedule_patch() -> bool:
    """Pin only the measured MP31 softmax-TopK shapes to one launch config."""

    if not _enabled():
        logger.info("MUSA TopK schedule disabled by %s", _ENV_NAME)
        return False

    device_name = _device_name()
    if "S5000" not in device_name.upper():
        logger.info("MUSA TopK schedule skipped on device %s", device_name)
        return False

    try:
        import triton
        from sglang.srt.hardware_backend.musa.kernels import topk as musa_topk
        from sglang.srt.layers.moe import topk as layer_topk
    except ImportError as exc:
        logger.warning("MUSA TopK schedule skipped: %s", exc)
        return False

    if getattr(musa_topk.topk_softmax, _PATCH_MARKER, False):
        return True

    selected_config = triton.Config({}, num_warps=1, num_stages=1)
    wrapped = _make_topk_wrapper(
        musa_topk.topk_softmax,
        musa_topk.topk_softmax_triton_kernel,
        selected_config,
    )
    musa_topk.topk_softmax = wrapped
    # This alias may already have been imported before vendor patches run.
    layer_topk.topk_softmax = wrapped
    logger.info(
        "MUSA S5000 softmax TopK schedule enabled for E=256, K=8 graph shapes "
        "and 1K-16K prefill: warps=1, stages=1"
    )
    return True
