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

"""Use MATE's pack-GQA FMHA path for the measured S5000 8K prefill shape."""

from __future__ import annotations

import importlib
import logging
import os
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)

_ENV_NAME = "SGLANG_MUSA_FMHA_PREFILL_PACK_GQA"
_PATCH_MARKER = "_sglang_fl_musa_fmha_prefill_pack_gqa"
_UNSET = object()
_match_logged = False


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


def _matches_s5000_prefill(
    m: int,
    head_ratio: int,
    headdim: int,
    headdim_v: int,
    element_size: int,
    enable_packgqa,
    has_qv: bool,
    is_fp8: bool,
    config,
) -> bool:
    return (
        _enabled()
        and "S5000" in _device_name().upper()
        and m == 8192
        and head_ratio == 8
        and headdim == headdim_v == 256
        and element_size == 2
        and enable_packgqa is None
        and not has_qv
        and not is_fp8
        and tuple(config[:4]) == (192, 64, 1, 1)
        and config[-1] is False
    )


def _wrap_get_fwd_kernel_config(original: Callable[..., Any]):
    if getattr(original, _PATCH_MARKER, False):
        return original

    @wraps(original)
    def wrapped(
        m,
        head_ratio,
        headdim,
        headdim_v,
        element_size,
        enable_packgqa=None,
        has_qv=False,
        is_fp8=False,
        is_high_regpressure=_UNSET,
    ):
        global _match_logged
        # MATE 0.2.7 adds this argument. Leave omitted arguments to the
        # original selector so older MATE versions still receive eight.
        extra = () if is_high_regpressure is _UNSET else (is_high_regpressure,)
        config = original(
            m,
            head_ratio,
            headdim,
            headdim_v,
            element_size,
            enable_packgqa,
            has_qv,
            is_fp8,
            *extra,
        )
        if is_high_regpressure is not _UNSET and is_high_regpressure:
            return config
        if not _matches_s5000_prefill(
            m,
            head_ratio,
            headdim,
            headdim_v,
            element_size,
            enable_packgqa,
            has_qv,
            is_fp8,
            config,
        ):
            return config

        tuned = tuple(config[:-1]) + (True,)
        if not _match_logged:
            logger.info(
                "MUSA S5000 FMHA prefill pack-GQA selected for "
                "q=8192, head_ratio=8, head_dim=256"
            )
            _match_logged = True
        return tuned

    setattr(wrapped, _PATCH_MARKER, True)
    return wrapped


def apply_musa_fmha_schedule_patch() -> bool:
    """Patch all MATE aliases of the forward config selector."""

    if not _enabled():
        logger.info("MUSA FMHA prefill pack-GQA patch disabled by %s", _ENV_NAME)
        return False
    if "S5000" not in _device_name().upper():
        return False

    try:
        fmha_utils = importlib.import_module("mate.jit.attention.fmha.fmha_utils")
        fmha_fwd = importlib.import_module("mate.jit.attention.fmha.fmha_fwd")
        fmha_metadata = importlib.import_module(
            "mate.jit.attention.fmha.fmha_get_metadata"
        )
    except ImportError as exc:
        logger.warning("MUSA FMHA prefill schedule patch skipped: %s", exc)
        return False

    wrapped = _wrap_get_fwd_kernel_config(fmha_utils._get_fwd_kernel_config)
    fmha_utils._get_fwd_kernel_config = wrapped
    fmha_fwd._get_fwd_kernel_config = wrapped
    fmha_metadata._get_metadata_kernel_config = wrapped
    logger.info("MUSA S5000 FMHA prefill pack-GQA schedule patch applied")
    return True
