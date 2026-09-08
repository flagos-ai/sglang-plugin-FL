# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Allow Triton 3.2 to scan disabled CUDA PDL branches on MUSA.

SGLang's portable kernels contain constexpr-guarded PDL calls. Triton 3.2
resolves their names before eliminating the false branch. Provide only the
two missing names; attempting to compile a live PDL call must still fail.
"""

from types import SimpleNamespace

import triton
import triton.language as tl


@triton.jit
def _unsupported_gdc():
    tl.static_assert(False, "CUDA PDL is not supported on MUSA")


def patch_triton_pdl_symbols():
    cuda = getattr(tl.extra, "cuda", None)
    if cuda is None:
        cuda = SimpleNamespace()
        tl.extra.cuda = cuda
    for name in ("gdc_wait", "gdc_launch_dependents"):
        if not hasattr(cuda, name):
            setattr(cuda, name, _unsupported_gdc)
