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

"""MetaX clamp_position guard: torch-native fallback when the JIT load fails.

``sglang.kernels.ops.attention.clamp_position`` compiles ``clamp_position``
with nvcc via ``load_jit``; MetaX torch is a CUDA alias (``is_cuda()`` True)
but ships no nvcc, so the JIT load raises. ``forward_batch_info`` binds
``clamp_position = clamp_position_cuda`` by name at module import; this patch
runs in ``load_plugin`` before that module imports, so rebinding the module
attribute here makes the by-name import pick up the guarded version.
"""

from __future__ import annotations

import logging
import sys

import torch

logger = logging.getLogger(__name__)

_patched = False


def patch_clamp_position() -> None:
    """Replace ``clamp_position_cuda`` with a JIT-load-guarded wrapper."""
    global _patched
    if _patched:
        return
    _patched = True

    try:
        from sglang.kernels.ops.attention import clamp_position as _cp
    except Exception as e:  # pragma: no cover - guard only
        logger.warning("Metax clamp_position patch skipped: %s", e)
        return

    original = _cp.clamp_position_cuda

    def clamp_position_cuda(seq_lens: torch.Tensor) -> torch.Tensor:
        """JIT clamp with a torch-native fallback for nvcc-less runtimes."""
        try:
            return original(seq_lens)
        except (RuntimeError, FileNotFoundError):
            # MetaX: CUDA-alias torch without nvcc makes the JIT load fail;
            # fall back to the same torch-native compute the non-CUDA branch
            # of forward_batch_info uses.
            return torch.clamp(seq_lens - 1, min=0)

    _cp.clamp_position_cuda = clamp_position_cuda

    # forward_batch_info binds clamp_position = clamp_position_cuda by name at
    # module import; if it is already loaded, rebind its reference too.
    fbi = sys.modules.get("sglang.srt.model_executor.forward_batch_info")
    if fbi is not None and hasattr(fbi, "clamp_position"):
        fbi.clamp_position = clamp_position_cuda

    logger.info("Metax clamp_position patch applied")
