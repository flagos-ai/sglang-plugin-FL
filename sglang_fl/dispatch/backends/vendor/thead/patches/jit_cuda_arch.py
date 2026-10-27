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

"""Use the arch-specific JIT target (e.g. ``sm_80a``) for sglang JIT kernels on PPU.

sglang builds JIT kernels for ``TVM_FFI_CUDA_ARCH_LIST=<major>.<minor>`` and
asserts ``__CUDA_ARCH__ == SGL_CUDA_ARCH`` in ``sgl_kernel/utils.cuh``. PPU's
nvcc compiles a plain ``sm_80`` target for both ppu001 (800) and ppu0015 (890),
so the second pass fails that assert. ``sm_80a`` only targets ppu001.

Set ``SGLANG_FL_THEAD_JIT_ARCH_SUFFIX`` to override the suffix ("" disables).
"""

import importlib
import logging
import os

logger = logging.getLogger(__name__)

# sglang <= 0.5.x: jit_kernel/utils.py; newer main: kernels/jit/utils/arch.py
_ARCH_MODULES = ("sglang.jit_kernel.utils", "sglang.kernels.jit.utils.arch")


def _patch_module(mod, suffix: str) -> None:
    orig_init = mod._init_jit_cuda_arch_once

    def _fix_suffix() -> None:
        arch = getattr(mod, "_CUDA_ARCH", None)
        if arch is not None and arch.major > 0 and not arch.suffix:
            mod._CUDA_ARCH = mod.ArchInfo(arch.major, arch.minor, suffix)

    def _init_with_suffix():
        orig_init()  # cached by sglang; only detects once
        _fix_suffix()

    mod._init_jit_cuda_arch_once = _init_with_suffix
    _fix_suffix()  # in case the arch was already detected before the plugin loaded


def patch_jit_cuda_arch() -> None:
    suffix = os.environ.get("SGLANG_FL_THEAD_JIT_ARCH_SUFFIX", "a")
    if not suffix:
        return
    for name in _ARCH_MODULES:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        if not hasattr(mod, "_init_jit_cuda_arch_once"):
            continue
        _patch_module(mod, suffix)
        logger.info("thead: sglang JIT CUDA arch suffix '%s' (%s)", suffix, name)
        return
    logger.warning("thead: sglang JIT arch module not found; JIT arch suffix not applied")
