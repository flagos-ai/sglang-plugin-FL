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

"""Kunlunxin: route is_cuda-gated sglang JIT-CUDA plumbing kernels to native.

Several sglang utilities pick between a CUDA kernel and a pure-torch fallback
at import time, gated on ``is_cuda() or is_hip()``.

On Kunlunxin ``is_cuda()`` is True (cuda-compat), so sglang selects the CUDA
branch. Those ``*_cuda`` variants JIT-compile with nvcc ``-std=c++20``;
Kunlun's host CUDA 11.7 nvcc caps at c++17, so compilation aborts.

sglang 0.5.18+: ``overlap_utils._resolve_future_token_ids*`` was removed
(FutureMap rewrite). Only ``forward_batch_info.clamp_position`` remains.
"""

import logging

logger = logging.getLogger(__name__)
_applied = False


def patch_clamp_position():
    """Rebind clamp_position kernel to torch-native on klx."""
    global _applied
    if _applied:
        return

    from sglang.srt.model_executor import forward_batch_info

    if not (
        hasattr(forward_batch_info, "clamp_position")
        and hasattr(forward_batch_info, "_clamp_position_native")
    ):
        _applied = True
        logger.warning(
            "Kunlunxin clamp_position patch is obsolete: sglang no longer "
            "exposes forward_batch_info.clamp_position/_clamp_position_native; "
            "remove patches/clamp_position.py"
        )
        return
        
    forward_batch_info.clamp_position = forward_batch_info._clamp_position_native

    _applied = True
    logger.info(
        "patched JIT-CUDA plumbing kernel -> torch-native "
        "(clamp_position; avoids nvcc -std=c++20 on Kunlunxin)"
    )