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

"""Robust fused-MoE scheduling for MTT S5000 decode and long prefill.

TorchAda's bundled Triton 3.2 decode configuration has a large performance
cliff on some S5000 systems with Triton 3.6.  A four-warp, K=64 configuration
is the validated replacement for the measured decode shape, and the measured
prefill schedules replace the generic tile for the long-prefill range and the
exact M=16384 shape.

Keep these as vendor monkeypatches: SGLang and TorchAda remain unmodified, and
operators can disable decode and prefill independently. Patch both resolvers
because SGLang v0.5.11 carries its own copy while some MUSA integration
versions call TorchAda's runtime copy directly.

See the MThreads plugin README section "Patch background and equivalence
notes" for the measured tile numbers and Triton version history.
"""

from __future__ import annotations

import importlib
import logging
import os
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)

_ENV_NAME = "SGLANG_MUSA_MOE_DECODE_SCHEDULE"
_PREFILL_ENV_NAME = "SGLANG_MUSA_MOE_PREFILL_SCHEDULE"
# Keep compiler backend optimization opt-in and confined to the measured M=64
# MUSA decode path.  The service candidate sets this to 1 explicitly.
_BACKEND_OPT_ENV_NAME = "SGLANG_MUSA_MOE_BACKEND_OPT"
_PATCH_MARKER = "_sglang_fl_musa_moe_schedule"
_decode_match_logged = False
_prefill_match_logged = False
_prefill_m16k_match_logged = False
_r2_m4_match_logged = False
# R2's core M4 BN64 selector requires this original SGLang baseline tile.
# The August image's TorchAda table instead selects BM128/G64 at M4.
# Explicit restoration only; do not change any other image table entry.
_R2_M4_BASELINE_ENV = "SGLANG_MUSA_M4_R2_BASELINE_SCHEDULE"
_R2_M4_BASELINE_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
}
_S5000_DECODE_CONFIG = {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 1,
}
_S5000_PREFILL_CONFIG = {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 4,
    "num_warps": 8,
    "num_stages": 1,
}
_S5000_PREFILL_M16K_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 4,
    "num_warps": 8,
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


def _prefill_enabled() -> bool:
    return os.environ.get(_PREFILL_ENV_NAME, "auto").strip().lower() not in {
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


def _backend_opt_enabled() -> bool:
    """Enable Triton backend optimization only for an explicit candidate opt-in."""

    if os.environ.get(_BACKEND_OPT_ENV_NAME, "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
        "enable",
        "enabled",
    }:
        return False
    if "S5000" not in _device_name().upper():
        return False
    try:
        import triton

        major, minor = (int(part) for part in str(triton.__version__).split(".", 2)[:2])
    except (AttributeError, ImportError, TypeError, ValueError):
        return False
    return (major, minor) == (3, 6)


def _matches_common_weight_contract(
    w1_shape,
    w2_shape,
    top_k: int,
    dtype,
    *,
    block_shape=None,
    per_channel_quant: bool = False,
) -> bool:
    """Shared measured weight, top-k, dtype and quantization contract.

    Every schedule selects the same Qwen3.6 TP2 BF16 expert shapes.  Only
    the token-count limits, the per-branch environment switches, the decode
    intermediate-size window, the backend-opt toggle and the M4-only
    ``is_marlin`` restriction remain branch-specific.
    """

    return (
        len(w1_shape) == 3
        and len(w2_shape) == 3
        and w1_shape[0] == w2_shape[0] == 256
        and w1_shape[2] == w2_shape[1] == 2048
        and w1_shape[1] == 2 * w2_shape[2]
        and top_k == 8
        and dtype is None
        and block_shape is None
        and not per_channel_quant
    )


def _on_s5000() -> bool:
    """Read the current device; deliberately not cached at install time."""

    return "S5000" in _device_name().upper()


def _select_musa_moe_schedule(
    w1_shape,
    w2_shape,
    top_k,
    dtype,
    M,
    is_marlin,
    block_shape,
    per_channel_quant,
) -> tuple[dict | None, dict, bool]:
    """Shared contract, then one explicit branch per measured schedule.

    Each branch returns ``(base_config, additions, baseline_down)`` where
    ``additions`` holds only its own options.  The caller copies ``base_config``
    and applies ``additions``, so branches never mutate or leak module-level
    constants.  No branch matches -> ``(None, {}, False)``.
    """

    global _decode_match_logged, _prefill_match_logged, _prefill_m16k_match_logged
    global _r2_m4_match_logged
    # Disabled or non-target calls must reach the original resolver without
    # inspecting its shape objects. Keep these checks per call: the runtime
    # switches can change after installation, including the independent M4 opt-in.
    m4_enabled = os.environ.get(_R2_M4_BASELINE_ENV, "0") == "1"
    if not (m4_enabled or _enabled() or _prefill_enabled()) or not _on_s5000():
        return None, {}, False
    if not _matches_common_weight_contract(
        w1_shape,
        w2_shape,
        top_k,
        dtype,
        block_shape=block_shape,
        per_channel_quant=per_channel_quant,
    ):
        return None, {}, False

    if (
        m4_enabled
        and w2_shape[2] == 256
        and M == 4
        and not is_marlin
    ):
        if not _r2_m4_match_logged:
            logger.info(
                "Restored R2 M4 baseline MoE tile BM16/BN32/BK64/G1; "
                "W13 BN64 remains a separate core opt-in"
            )
            _r2_m4_match_logged = True
        # R2 baseline has no independent down schedule. The core uses the
        # unchanged baseline config for W2, and copies W13 for BN64.
        return _R2_M4_BASELINE_CONFIG, {}, True
    if _enabled() and w2_shape[2] in (256, 512) and M == 64:
        backend_opt = w2_shape[2] == 256 and _backend_opt_enabled()
        additions = {"enable_backend_opt": True} if backend_opt else {}
        if not _decode_match_logged:
            logger.info(
                "MUSA S5000 MoE decode schedule selected for "
                "w1=%s, w2=%s, top_k=%s, M=%s, backend_opt=%s",
                tuple(w1_shape),
                tuple(w2_shape),
                top_k,
                M,
                backend_opt,
            )
            _decode_match_logged = True
        return _S5000_DECODE_CONFIG, additions, False
    if _prefill_enabled() and w2_shape[2] == 256 and 2048 <= M <= 8192:
        if not _prefill_match_logged:
            logger.info(
                "MUSA S5000 MoE prefill schedule selected for "
                "w1=%s, w2=%s, top_k=%s, M=%s",
                tuple(w1_shape),
                tuple(w2_shape),
                top_k,
                M,
            )
            _prefill_match_logged = True
        return _S5000_PREFILL_CONFIG, {}, False
    if _prefill_enabled() and w2_shape[2] == 256 and M == 16384:
        if not _prefill_m16k_match_logged:
            logger.info(
                "MUSA S5000 MoE exact M=16384 prefill schedule selected for "
                "w1=%s, w2=%s, top_k=%s",
                tuple(w1_shape),
                tuple(w2_shape),
                top_k,
            )
            _prefill_m16k_match_logged = True
        return _S5000_PREFILL_M16K_CONFIG, {}, False
    return None, {}, False


def _schedule_result(
    base_config: dict,
    additions: dict,
    baseline_down: bool,
    return_down_config: bool,
):
    """Copy the selected module config and build the caller's return shape."""

    config = dict(base_config)
    config.update(additions)
    if not return_down_config:
        return config
    if baseline_down:
        # M4 owns its return contract explicitly; the generic down assembly
        # must not substitute a real down config here.
        return config, (None, None)
    return config, (dict(config), config["BLOCK_SIZE_M"])


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
        base_config, additions, baseline_down = _select_musa_moe_schedule(
            w1_shape,
            w2_shape,
            top_k,
            dtype,
            M,
            is_marlin,
            block_shape,
            per_channel_quant,
        )
        if base_config is None:
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
        return _schedule_result(
            base_config, additions, baseline_down, return_down_config
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

    if not (_enabled() or _prefill_enabled()):
        logger.info(
            "MUSA S5000 MoE schedule patch disabled by %s and %s",
            _ENV_NAME,
            _PREFILL_ENV_NAME,
        )
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
        "MUSA S5000 MoE schedules applied: decode M=64 "
        "(M32/N32/K64/G1/W4/S1); prefill M=2048..8192 "
        "(M32/N128/K64/G4/W8/S1); exact M=16384 "
        "(M64/N128/K64/G4/W8/S1)"
    )
    return True
