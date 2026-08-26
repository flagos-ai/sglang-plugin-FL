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

"""Robust fused-MoE scheduling for MTT S5000 decode.

TorchAda's bundled Triton 3.2 configuration uses eight warps and a K=128
tile for the Qwen3.6-35B-A3B TP2 decode shape.  That configuration has a
large performance cliff on some S5000 systems with Triton 3.6.  A four-warp,
K=64 configuration is within a few percent of the old-system optimum and is
about three times faster on the affected systems.

Keep this as a vendor monkeypatch: SGLang and TorchAda remain unmodified, and
operators can disable it with ``SGLANG_MUSA_MOE_DECODE_SCHEDULE=off``.  Patch
both resolvers because SGLang v0.5.11 carries its own copy while some MUSA
integration versions call TorchAda's runtime copy directly.
"""

from __future__ import annotations

import importlib
import logging
import os
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)

_ENV_NAME = "SGLANG_MUSA_MOE_DECODE_SCHEDULE"
_PATCH_MARKER = "_sglang_fl_musa_moe_schedule"
_match_logged = False
_S5000_DECODE_CONFIG = {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 1,
}


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

        if hasattr(torch, "musa") and torch.musa.is_available():
            return str(torch.musa.get_device_name())
    except Exception:
        pass
    return ""


def _matches_s5000_decode(
    w1_shape,
    w2_shape,
    top_k: int,
    dtype,
    M: int,
    *,
    block_shape=None,
    per_channel_quant: bool = False,
) -> bool:
    """Match only the measured Qwen3.6-35B-A3B TP2 BF16 decode shape."""

    return (
        _enabled()
        and "S5000" in _device_name().upper()
        and len(w1_shape) == 3
        and len(w2_shape) == 3
        and w1_shape[0] == w2_shape[0] == 256
        and w1_shape[2] == w2_shape[1] == 2048
        and w1_shape[1] == 2 * w2_shape[2]
        and w2_shape[2] in (256, 512)
        and top_k == 8
        and dtype is None
        and M == 64
        and block_shape is None
        and not per_channel_quant
    )


def _wrap_try_get_optimal_moe_config(original: Callable[..., Any]):
    if getattr(original, _PATCH_MARKER, False):
        return original

    @wraps(original)
    def wrapped(
        w1_shape,
        w2_shape,
        top_k,
        dtype,
        M,
        is_marlin=False,
        block_shape=None,
        per_channel_quant=False,
        return_down_config=False,
    ):
        global _match_logged
        if _matches_s5000_decode(
            w1_shape,
            w2_shape,
            top_k,
            dtype,
            M,
            block_shape=block_shape,
            per_channel_quant=per_channel_quant,
        ):
            config = dict(_S5000_DECODE_CONFIG)
            if not _match_logged:
                logger.info(
                    "MUSA S5000 MoE decode schedule selected for "
                    "w1=%s, w2=%s, top_k=%s, M=%s",
                    tuple(w1_shape),
                    tuple(w2_shape),
                    top_k,
                    M,
                )
                _match_logged = True
            if return_down_config:
                return config, (dict(config), config["BLOCK_SIZE_M"])
            return config
        return original(
            w1_shape,
            w2_shape,
            top_k,
            dtype,
            M,
            is_marlin=is_marlin,
            block_shape=block_shape,
            per_channel_quant=per_channel_quant,
            return_down_config=return_down_config,
        )

    setattr(wrapped, _PATCH_MARKER, True)
    return wrapped


def _patch_resolver(config_module_name: str, fused_moe_module_name: str) -> bool:
    try:
        config_module = importlib.import_module(config_module_name)
        fused_moe_module = importlib.import_module(fused_moe_module_name)
    except ImportError as exc:
        logger.debug("MUSA MoE resolver %s unavailable: %s", config_module_name, exc)
        return False

    wrapped = _wrap_try_get_optimal_moe_config(
        config_module.try_get_optimal_moe_config
    )
    config_module.try_get_optimal_moe_config = wrapped
    fused_moe_module.try_get_optimal_moe_config = wrapped
    return True


def apply_musa_moe_schedule_patch() -> bool:
    """Patch SGLang/TorchAda config resolvers and already-imported aliases."""

    if not _enabled():
        logger.info("MUSA S5000 MoE decode schedule patch disabled by %s", _ENV_NAME)
        return False

    patched = False
    patched |= _patch_resolver(
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config",
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe",
    )
    patched |= _patch_resolver(
        "torchada.triton.runtime.fused_moe.config",
        "torchada.triton.runtime.fused_moe.fused_moe",
    )
    if not patched:
        logger.warning("MUSA S5000 MoE decode schedule patch skipped: no resolver")
        return False

    logger.info(
        "MUSA S5000 MoE decode schedule patch applied: M=64, "
        "BLOCK_M=32, BLOCK_N=32, BLOCK_K=64, warps=4, stages=1"
    )
    return True
