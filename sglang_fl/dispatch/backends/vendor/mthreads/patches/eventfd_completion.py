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

"""Replace MUSA output Event waits with native callback + eventfd waits."""

from __future__ import annotations

import logging
import os
from functools import wraps

logger = logging.getLogger(__name__)

_ENV_NAME = "SGLANG_MUSA_EVENTFD_COMPLETION"
_PATCH_MARKER = "_sglang_fl_musa_eventfd_completion"


def _device_is_s5000() -> bool:
    try:
        import torch

        return "S5000" in str(torch.musa.get_device_name()).upper()
    except Exception:
        return False


def _enabled() -> bool:
    value = os.environ.get(_ENV_NAME, "auto").strip().lower()
    if value in ("0", "false", "no", "off"):
        return False
    if value in ("1", "true", "yes", "on"):
        return True
    if value != "auto":
        raise ValueError(f"Unsupported {_ENV_NAME} value: {value!r}")
    return _device_is_s5000()


def _wrap_copy_to_cpu(original):
    if getattr(original, _PATCH_MARKER, False):
        return original

    @wraps(original)
    def wrapped(result, return_logprob: bool):
        next_token_ids = getattr(result, "next_token_ids", None)
        copy_device = getattr(next_token_ids, "device", None)
        value = original(result, return_logprob)
        fallback_event = getattr(result, "copy_done", None)
        if copy_device is None or fallback_event is None:
            return value
        from ..eventfd_completion import (
            MusaCompletionEventProxy,
            try_enqueue_musa_completion,
        )

        completion = try_enqueue_musa_completion(copy_device)
        if completion is not None:
            result.copy_done = MusaCompletionEventProxy(
                completion, fallback_event
            )
        return value

    setattr(wrapped, _PATCH_MARKER, True)
    return wrapped


def apply_musa_eventfd_completion_patch() -> bool:
    if not _enabled():
        logger.info("MUSA eventfd completion disabled by %s", _ENV_NAME)
        return False
    try:
        from sglang.srt.managers import utils as manager_utils
    except Exception as exc:
        logger.warning("MUSA eventfd completion patch skipped: %s", exc)
        return False

    original = manager_utils.GenerationBatchResult.copy_to_cpu
    if getattr(original, _PATCH_MARKER, False):
        return True
    manager_utils.GenerationBatchResult.copy_to_cpu = _wrap_copy_to_cpu(original)
    logger.info(
        "MUSA eventfd output completion enabled with Event fallback "
        "(pool size=%s)",
        os.environ.get("SGLANG_MUSA_EVENTFD_COMPLETION_POOL_SIZE", "64"),
    )
    return True


__all__ = ["apply_musa_eventfd_completion_patch"]
