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

"""Vendor monkey-patches on sglang internals for MetaX — entrypoint.

MetaX torch is a CUDA alias (``torch.version.cuda == "11.6"``, ``is_cuda()``
True) that ships no nvcc, so CUDA branches that load JIT kernels or vendor
flashinfer-only symbols would crash on import. These patches replace the
per-vendor edits that used to be applied to the wheel build:

  - clamp_position: guard the JIT load with a torch-native fallback
  - vision:         preset cudnn_batch_prefill_with_kv_cache to None
  - fp8_utils:      preset bmm_fp8 to None

The sglang source tree stays pristine; the patches are applied at load_plugin
time, before sglang model modules import (mirrors the ascend vendor layout).
"""

import logging

from .patches.clamp_position import patch_clamp_position
from .patches.fp8_utils import patch_fp8_bmm_guard
from .patches.vision import patch_vision_cudnn_guard

logger = logging.getLogger(__name__)

_patches_applied = False


def apply_metax_patches() -> None:
    """Apply all MetaX-specific patches."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    patch_clamp_position()
    patch_vision_cudnn_guard()
    patch_fp8_bmm_guard()


apply_metax_patches()
