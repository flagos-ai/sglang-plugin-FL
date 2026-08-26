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

"""Keep text-only decode MRoPE positions on the MUSA device.

SGLang v0.5.11 builds a ``[3, batch]`` CPU tensor on every decode step and
copies it to the accelerator.  The copy is only 1.5 KiB at batch size 64, but
MUSA's pageable-memory ``musaMemcpyAsync`` waits for the preceding graph and
blocks the scheduler thread.  Text-only MRoPE axes are identical, so a
zero-stride view of the already resident decode positions is equivalent and
avoids both the allocation and the host-to-device transfer.
"""

from __future__ import annotations

import logging
import os
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)

_ENV_NAME = "SGLANG_MUSA_DECODE_MROPE_DEVICE_POSITIONS"
_PATCH_MARKER = "_sglang_fl_musa_mrope_device_positions"


def _enabled() -> bool:
    return os.environ.get(_ENV_NAME, "auto").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "disable",
        "disabled",
    }


def _can_reuse_device_positions(
    forward_batch: Any,
    model_worker_batch: Any,
    *,
    rl_on_policy_target: Any,
) -> bool:
    """Return whether decode MRoPE is exactly ``positions.expand(3, -1)``."""

    try:
        is_decode = forward_batch.forward_mode.is_decode()
        multimodal_inputs = model_worker_batch.multimodal_inputs
    except AttributeError:
        return False

    return bool(
        is_decode
        and (
            rl_on_policy_target is not None
            or all(mm_input is None for mm_input in multimodal_inputs)
        )
    )


def _wrap_compute_mrope_positions(
    original: Callable[..., Any], get_global_server_args: Callable[[], Any]
):
    if getattr(original, _PATCH_MARKER, False):
        return original

    @wraps(original)
    def wrapped(self, model_runner, batch):
        if _enabled() and _can_reuse_device_positions(
            self,
            batch,
            rl_on_policy_target=get_global_server_args().rl_on_policy_target,
        ):
            self.mrope_positions = self.positions.unsqueeze(0).expand(3, -1)
            return None
        return original(self, model_runner, batch)

    setattr(wrapped, _PATCH_MARKER, True)
    return wrapped


def apply_musa_mrope_device_positions_patch() -> bool:
    """Monkeypatch SGLang's decode MRoPE producer without modifying SGLang."""

    if not _enabled():
        logger.info("MUSA decode MRoPE device positions disabled by %s", _ENV_NAME)
        return False

    try:
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.server_args import get_global_server_args
    except ImportError as exc:
        logger.warning("MUSA decode MRoPE device positions skipped: %s", exc)
        return False

    original = ForwardBatch._compute_mrope_positions
    wrapped = _wrap_compute_mrope_positions(original, get_global_server_args)
    if wrapped is original:
        return True

    ForwardBatch._compute_mrope_positions = wrapped
    logger.info("MUSA decode MRoPE device positions patch applied")
    return True
