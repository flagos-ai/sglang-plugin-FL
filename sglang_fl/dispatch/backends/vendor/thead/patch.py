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

"""T-Head PPU monkey-patches on sglang internals — entrypoint.

Auto-imported by ``sglang_fl.load_plugin()`` (see ``_apply_vendor_patches``).
"""

from .patches.graph_capture_status import patch_graph_capture_status
from .patches.jit_cuda_arch import patch_jit_cuda_arch
from .patches.moe_flaggems import patch_moe_flaggems
from .patches.sgl_kernel_compat import patch_sgl_kernel_compat

_patches_applied = False


def apply_thead_patches() -> None:
    """Apply all PPU-specific patches once."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    patch_sgl_kernel_compat()
    patch_jit_cuda_arch()
    patch_moe_flaggems()
    patch_graph_capture_status()


apply_thead_patches()
