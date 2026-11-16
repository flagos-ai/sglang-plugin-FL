# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Use SGLang's request-pool fallback on the affected Ascend compiler stack."""

from __future__ import annotations

import importlib
import logging

from ..runtime import uses_flagtree_cann921

logger = logging.getLogger(__name__)

_ALLOCATION_MODULE = "sglang.srt.mem_cache.allocation"


def _request_pool_triton_unavailable(_backend: str) -> bool:
    return False


def patch_request_pool_write_fallback() -> bool:
    """Select the upstream PyTorch branch of `write_cache_indices`."""

    if not uses_flagtree_cann921():
        return False
    try:
        allocation = importlib.import_module(_ALLOCATION_MODULE)
        support_triton = allocation.support_triton
    except (AttributeError, ImportError):
        logger.debug("SGLang request-pool writer is unavailable", exc_info=True)
        return False
    if support_triton is _request_pool_triton_unavailable:
        return True

    # SGLang 0.5.18 uses this module-local alias only in write_cache_indices.
    # FlagTree cannot compile its indirect pointer load on CANN 9.2.1.
    # https://github.com/flagos-ai/FlagTree/issues/1668
    allocation.support_triton = _request_pool_triton_unavailable
    logger.info("Using SGLang's PyTorch fallback for Ascend request-pool writes")
    return True
