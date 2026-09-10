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

"""Iluvatar: route the is_cuda-gated clamp_position JIT kernel to native.

``forward_batch_info`` picks its clamp_position implementation at import time:

    if is_cuda() or is_hip():
        from ...clamp_position import clamp_position_cuda   # nvcc JIT
    else:
        clamp_position = _clamp_position_native             # pure torch

corex torch is CUDA-alias, so `is_cuda()` is True and the JIT variant is
selected. That variant compiles a .cu file on every new (dtype) combination
with the host nvcc, which corex does not ship:

    RuntimeError: Failed to build JIT module
    sgl_kernel_jit_clamp_position_int64_t in /root/.cache/sglang/jit/sm71/...

It is called from ForwardBatch.init_new on every batch, so serve dies at the
first request. The module already ships the pure-torch equivalent; rebind the
module global to it. Call sites look the global up at call time, so the rebind
takes effect regardless of import order (same seam as the kunlunxin patch).

Only clamp_position needs this on 0.5.18 — the overlap-scheduler plumbing the
kunlunxin patch also rebinds no longer exists under that name.
"""

import logging

logger = logging.getLogger(__name__)
_applied = False


def patch_clamp_position():
    """Rebind the JIT clamp_position to its torch-native twin on corex."""
    global _applied
    if _applied:
        return

    from sglang.srt.model_executor import forward_batch_info

    forward_batch_info.clamp_position = forward_batch_info._clamp_position_native

    _applied = True
    logger.info(
        "iluvatar: clamp_position -> torch-native "
        "(corex ships no nvcc for the JIT variant)"
    )
