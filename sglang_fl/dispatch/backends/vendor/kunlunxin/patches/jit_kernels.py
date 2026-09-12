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

"""Kunlunxin: make sglang's JIT CUDA predicates answer False.

Several `can_use_*` predicates decide whether to use a JIT-compiled CUDA kernel
by compiling it and reporting success:

    def can_use_x(...):
        try:
            _jit_x_module(...)   # nvcc, never launched
            return True
        except Exception:
            return False

The build targets the capability the device reports, and kunlunxin's
cuda-compat device answers (8, 6), so nvcc emits sm_86 SASS. Compilation
succeeds; the failure lands at the launch, far from the decision that was
wrong:

    RuntimeError: ... qknorm.cuh:173: CUDA error: invalid device function
    RuntimeError: ... kvcache.cuh:316: CUDA error: invalid device function

"invalid device function" is the driver reporting that no SASS in the image
matches this device. sm_86 is an NVIDIA architecture and this device executes
its own instruction set, so no nvcc-produced SASS can run here — the toolchain
being a real nvcc is exactly why nothing notices.

Each entry below names one predicate and the import sites that copy it. The
predicates are rebound in their defining module and in every module that
already did a from-import (the name is copied at import time, so patching only
the definition would miss a caller that imported first). A module not yet
imported needs nothing: it picks the rebind up from the definition.

Add an entry here when a new kernel proves unrunnable on this device; the
predicate's own `except` path is what routes the caller to native.
"""

import logging
import sys

logger = logging.getLogger(__name__)
_applied = False

# (defining module, attribute) -> module paths that from-import the same name.
# The attribute name differs where a consumer aliases it (vision.py).
_PREDICATES = (
    (
        ("sglang.kernels.ops.layernorm.norm", "can_use_fused_inplace_qknorm"),
        (
            ("sglang.srt.models.utils", "can_use_fused_inplace_qknorm"),
            ("sglang.srt.layers.attention.vision", "can_use_jit_qk_norm"),
        ),
    ),
    (
        ("sglang.kernels.ops.kvcache.kvcache", "can_use_store_cache"),
        (("sglang.srt.mem_cache.memory_pool", "can_use_store_cache"),),
    ),
)


def patch_jit_kernel_predicates():
    """Force the named JIT-CUDA predicates to False on kunlunxin."""
    global _applied
    if _applied:
        return
    _applied = True

    def _always_false(*args, **kwargs) -> bool:
        return False

    for (def_module, def_name), consumers in _PREDICATES:
        try:
            module = __import__(def_module, fromlist=[def_name])
            setattr(module, def_name, _always_false)
        except Exception as e:
            logger.warning("kunlunxin: cannot rebind %s.%s: %r", def_module, def_name, e)
            continue
        for mod_path, attr in consumers:
            mod = sys.modules.get(mod_path)
            if mod is not None and hasattr(mod, attr):
                setattr(mod, attr, _always_false)

    logger.info(
        "kunlunxin: JIT CUDA kernel predicates forced False "
        "(nvcc emits sm_86 SASS, which this device cannot execute)"
    )
