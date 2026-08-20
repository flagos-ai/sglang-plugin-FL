# Copyright 2026 BAAI. All rights reserved.
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

"""Import-time stub of the CUDA-only ``sgl_kernel`` package.

sglang imports some ``sgl_kernel`` symbols at module level even on devices
without a CUDA sgl-kernel build. On those devices the kernels are never
supposed to run: sglang-plugin-FL's dispatch layer intercepts the fused
kernels and routes them to flagos (FlagGems) / vendor / reference backends.
This module keeps the imports alive and turns any actual kernel call into a
loud error instead of a silent no-op.
"""

_SUBMODULES = (
    "allreduce",
    "elementwise",
    "flash_attn",
    "flash_mla",
    "kvcacheio",
    "mamba",
    "sparse_flash_attn",
)


def _make_raiser(name: str):
    def _raiser(*args, **kwargs):
        raise NotImplementedError(
            f"sgl_kernel.{name} is a CUDA kernel stub; this platform must route "
            "the op through sglang_fl dispatch (flagos/vendor/reference). If you "
            "see this error, a fused-kernel call was NOT intercepted by the "
            "plugin dispatch layer."
        )

    return _raiser


def __getattr__(name: str):
    if name in ("__path__", "__spec__", "__all__", "__file__"):
        raise AttributeError(name)
    return _make_raiser(name)
