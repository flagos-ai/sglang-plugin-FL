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

"""MTT S5000 launch policy for SGLang's packed GDN decode kernel."""

from __future__ import annotations

import importlib
import logging
import os

import torch

from sglang_fl.patches.triton_kernel import patch_kernel_launch_meta

logger = logging.getLogger(__name__)

_KERNEL_MODULE = "sglang.srt.layers.attention.fla.fused_recurrent"
_KERNEL_NAME = "fused_recurrent_gated_delta_rule_packed_decode_kernel"
_ENV = "SGLANG_FL_MUSA_GDN_PACKED_DECODE_NUM_WARPS"
_LEGACY_ENV = "SGLANG_MUSA_GDN_PACKED_DECODE_NUM_WARPS"
_VALID_NUM_WARPS = {1, 2, 4, 8, 16}
_DEVICE_DEFAULTS = {"S5000": 8}


def _get_musa_device_name() -> str | None:
    try:
        musa = getattr(torch, "musa", None)
        if musa is None or not musa.is_available() or musa.device_count() < 1:
            return None
        return str(musa.get_device_name(0))
    except Exception as exc:
        logger.warning("MUSA device-name query failed; GDN tuning skipped: %s", exc)
        return None


def _platform_default(device_name: str | None) -> int | None:
    normalized = (device_name or "").upper()
    for model, num_warps in _DEVICE_DEFAULTS.items():
        if model in normalized:
            return num_warps
    return None


def _resolve_num_warps(device_name: str | None) -> int | None:
    raw = os.environ.get(_ENV)
    source = _ENV
    if raw is None:
        raw = os.environ.get(_LEGACY_ENV)
        source = _LEGACY_ENV

    if raw is None or raw.strip().lower() == "auto":
        return _platform_default(device_name)
    if raw.strip().lower() in {"0", "off", "false", "disable", "disabled"}:
        return None

    try:
        num_warps = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{source} must be auto/off/1/2/4/8/16, got {raw!r}"
        ) from exc
    if num_warps not in _VALID_NUM_WARPS:
        raise ValueError(f"{source} must be auto/off/1/2/4/8/16, got {raw!r}")
    return num_warps


def apply_musa_gdn_launch_patch() -> None:
    """Apply the S5000-tuned packed-decode launch policy without editing SGLang."""

    device_name = _get_musa_device_name()
    num_warps = _resolve_num_warps(device_name)
    if num_warps is None:
        logger.info(
            "MUSA packed GDN decode launch tuning skipped for device=%s",
            device_name or "unknown",
        )
        return

    try:
        module = importlib.import_module(_KERNEL_MODULE)
        patch_kernel_launch_meta(module, _KERNEL_NAME, {"num_warps": num_warps})
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "MUSA packed GDN decode launch tuning is unavailable in this "
            "SGLang revision: %s",
            exc,
        )
        return
    logger.info(
        "Patched MUSA packed GDN decode launch meta for device=%s: num_warps=%d",
        device_name or "unknown",
        num_warps,
    )
