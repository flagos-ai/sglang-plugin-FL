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

"""Ascend patches that must run before SGLang imports its layers.

Auto-imported by ``sglang_fl.activate_platform()`` (see
``_apply_vendor_early_patches``). Keep imports here free of ``sglang.srt``.
"""

import logging

from .import_guard import install_import_guards

logger = logging.getLogger(__name__)

install_import_guards()

# The GDN backend launches ``torch.ops.npu.causal_conv1d`` on every step. The
# operations Runtime ships no wheel that registers it, so register the plain
# PyTorch fallback while the placeholder is active. No-op when a real
# sgl_kernel_npu (or the previous CI image's vendor wheel) already did.
try:
    from .sgl_kernel_npu_fallbacks import register_causal_conv1d

    if "sgl_kernel_npu" in install_import_guards():
        if register_causal_conv1d():
            logger.info("ascend: npu::causal_conv1d fallback registered")
except Exception:  # pragma: no cover - torch not importable in this process
    logger.debug("ascend: torch op registration skipped", exc_info=True)
