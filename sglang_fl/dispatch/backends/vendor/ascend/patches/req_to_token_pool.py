# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Use SGLang's request-pool fallback on the affected Ascend compiler stack."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from importlib import metadata
from pathlib import Path

logger = logging.getLogger(__name__)

_ALLOCATION_MODULE = "sglang.srt.mem_cache.allocation"


def _is_affected_runtime() -> bool:
    try:
        if metadata.version("sglang") != "0.5.18":
            return False
        flagtree = metadata.distribution("flagtree")
    except metadata.PackageNotFoundError:
        return False
    if flagtree.version != "0.7.0+ascend3.5":
        return False

    triton = importlib.util.find_spec("triton")
    owned_triton = Path(flagtree.locate_file("triton/__init__.py")).resolve()
    if (
        triton is None
        or triton.origin is None
        or Path(triton.origin).resolve() != owned_triton
    ):
        return False

    toolkit = Path(
        os.environ.get("ASCEND_HOME_PATH")
        or os.environ.get("ASCEND_HOME")
        or os.environ.get("ASCEND_TOOLKIT_HOME")
        or "/usr/local/Ascend/ascend-toolkit/latest"
    )
    for directory in ("aarch64-linux", "arm64-linux", "."):
        for filename in ("ascend_all_cann_install.info", "ascend_toolkit_install.info"):
            info = toolkit / directory / filename
            try:
                lines = info.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line in lines:
                name, separator, value = line.partition("=")
                if separator and name.strip().lower() == "version":
                    return value.strip().strip("\"'") == "9.2.1"
    return False


def _request_pool_triton_unavailable(_backend: str) -> bool:
    return False


def patch_request_pool_write_fallback() -> bool:
    """Select the upstream PyTorch branch of `write_cache_indices`."""

    if not _is_affected_runtime():
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
