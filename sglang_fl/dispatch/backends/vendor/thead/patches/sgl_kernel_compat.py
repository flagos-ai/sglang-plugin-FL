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

"""Provide the SGLang symbols that PPU builds of sgl_kernel expect.

PPU's sgl_kernel (``flash_attn.py``, ``flash_mla.py``) was built against
T-Head's patched SGLang and reads symbols upstream SGLang does not define:

- ``sglang.srt.utils.is_ppu``
- ``envs.SGLANG_PROFILE_NVTX`` / ``envs.SGLANG_PROFILE_NVTX_PRINT_SEQLEN``

SGLang 0.5.18's EAGLE/MTP worker imports ``flashmla_backend`` (and with it
``sgl_kernel.flash_mla``) while capturing draft CUDA graphs, so without these
MTP with CUDA graphs fails on PPU. Definitions match T-Head's SGLang:
``is_ppu()`` is True on a PPU host (this patch only runs on thead) and the NVTX
switches are booleans defaulting to False. Existing definitions are kept.
"""

import importlib
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

_UTILS_MODULES = ("sglang.srt.utils.common", "sglang.srt.utils")
_ENV_BOOLS = ("SGLANG_PROFILE_NVTX", "SGLANG_PROFILE_NVTX_PRINT_SEQLEN")


@lru_cache(maxsize=1)
def is_ppu() -> bool:
    return True


def _patch_is_ppu() -> None:
    for name in _UTILS_MODULES:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        if not hasattr(mod, "is_ppu"):
            mod.is_ppu = is_ppu


def _patch_envs() -> None:
    try:
        from sglang.srt.environ import EnvBool, Envs
    except ImportError:
        return
    for name in _ENV_BOOLS:
        if hasattr(Envs, name):
            continue
        # Fields are plain class attributes named via __set_name__ when the
        # registry class is built; set the name directly for late additions.
        field = EnvBool(False)
        field.name = name
        setattr(Envs, name, field)


def patch_sgl_kernel_compat() -> None:
    _patch_is_ppu()
    _patch_envs()
    logger.info("thead: sglang compat symbols provided for PPU sgl_kernel")
