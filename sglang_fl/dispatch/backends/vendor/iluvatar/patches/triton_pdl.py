# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Let vendor Triton scan disabled PDL branches while rejecting live PDL."""

import importlib
import logging
import sys
from types import SimpleNamespace

import triton
import triton.language as tl

logger = logging.getLogger(__name__)
_applied = False


@triton.jit
def _unsupported_gdc():
    tl.static_assert(False, "CUDA PDL is not supported on Iluvatar CoreX")


def _no_pdl() -> bool:
    return False


def patch_triton_pdl_intrinsics() -> None:
    """Resolve missing names in dead branches and disable the capability query."""
    global _applied
    if _applied:
        return

    # This imports only the installed vendor/shim package, never SGLang.
    # Set the real bool before SGLang imports and caches the capability query.
    utils = importlib.import_module("sgl_kernel.utils")
    original = getattr(utils, "is_arch_support_pdl", None)
    utils.is_arch_support_pdl = _no_pdl
    if original is not None:
        for name, module in tuple(sys.modules.items()):
            if (
                name.startswith("sglang.")
                and getattr(module, "is_arch_support_pdl", None) is original
            ):
                module.is_arch_support_pdl = _no_pdl

    cuda = getattr(tl.extra, "cuda", None)
    if cuda is None:
        cuda = SimpleNamespace()
        tl.extra.cuda = cuda
    added = []
    for name in ("gdc_wait", "gdc_launch_dependents"):
        if not hasattr(cuda, name):
            setattr(cuda, name, _unsupported_gdc)
            added.append(name)

    _applied = True
    logger.info("Iluvatar: PDL disabled; added missing Triton symbols %s", added)
