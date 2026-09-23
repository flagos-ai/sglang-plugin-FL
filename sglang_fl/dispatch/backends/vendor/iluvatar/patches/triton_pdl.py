# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Let Triton 3.2 scan disabled CUDA PDL branches on iluvatar corex.

SGLang 0.5.18 portable kernels contain constexpr-guarded PDL calls.
Triton 3.2 resolves those names before eliminating the false branch.
Triton 3.6 already exports the symbols, so this is a no-op there.
"""

from types import SimpleNamespace

import triton
import triton.language as tl


@triton.jit
def _unsupported_gdc():
    tl.static_assert(False, "CUDA PDL is not supported on iluvatar corex")


def patch_triton_pdl_symbols():
    cuda = getattr(tl.extra, "cuda", None)
    if cuda is None:
        cuda = SimpleNamespace()
        tl.extra.cuda = cuda
    for name in ("gdc_wait", "gdc_launch_dependents"):
        if not hasattr(cuda, name):
            setattr(cuda, name, _unsupported_gdc)
