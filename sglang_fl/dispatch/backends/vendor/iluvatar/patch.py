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

"""Vendor monkey-patches on sglang internals — entrypoint.

Auto-imported by ``sglang_fl.load_plugin()`` (see ``_apply_vendor_patches``).
Add one ``patch_xxx`` call per concern; put the implementation under ``patches/``.

Iluvatar is CUDA-alias: `is_cuda()` is True and `torch.cuda` works, so sglang's
CUDA paths are selected even though the corex platform has neither an NVIDIA
device nor the NVIDIA-only packages those paths expect. Each patch here closes
one such gap; see the module docstrings for the individual failure.
"""

import logging

from .patches.clamp_position import patch_clamp_position
from .patches.legacy_gpu_gate import patch_legacy_gpu_gate
from .patches.triton_pdl import patch_triton_pdl_intrinsics

logger = logging.getLogger(__name__)
_patches_applied = False


def apply_iluvatar_patches() -> None:
    """Apply all iluvatar-specific patches."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    patch_clamp_position()
    patch_legacy_gpu_gate()
    patch_triton_pdl_intrinsics()


apply_iluvatar_patches()
