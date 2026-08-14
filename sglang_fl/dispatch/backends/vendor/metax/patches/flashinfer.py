"""Patch FlashInfer communication support on MetaX environments."""

from __future__ import annotations

import importlib
import logging
import sys
import types

logger = logging.getLogger(__name__)

_FLASHINFER_COMM_MODULE = "flashinfer.comm"
_FLASHINFER_MNNVL_MODULE = "flashinfer.comm.mnnvl"
_MNNVL_PATCHED_MARKER = "__sglang_fl_metax_flashinfer_mnnvl_patched__"


def _get_or_create_module(module_name: str, package: str) -> types.ModuleType:
    module = sys.modules.get(module_name)
    if isinstance(module, types.ModuleType):
        return module

    try:
        module = importlib.import_module(module_name)
        if isinstance(module, types.ModuleType):
            return module
    except Exception:
        pass

    module = types.ModuleType(module_name)
    module.__loader__ = None
    module.__file__ = "<sglang_fl_metax_patch>"
    module.__package__ = package
    sys.modules[module_name] = module
    return module



def patch_flashinfer_utils_comm_backend() -> None:
    """Patch flashinfer.comm.mnnvl while preserving flashinfer_utils.

    ``flashinfer_utils`` later subclasses ``CommBackend``, so this symbol cannot
    be ``None``. We preserve the SGLang module and only provide the placeholder
    base class that the local source edit would define.
    """
    flashinfer = _get_or_create_module("flashinfer", "flashinfer")
    comm_module = _get_or_create_module(_FLASHINFER_COMM_MODULE, "flashinfer")
    mnnvl_module = _get_or_create_module(_FLASHINFER_MNNVL_MODULE, "flashinfer.comm")
    setattr(flashinfer, "comm", comm_module)
    setattr(comm_module, "mnnvl", mnnvl_module)

    class CommBackend:
        """Placeholder base class when flashinfer mnnvl is unavailable."""

        pass

    setattr(mnnvl_module, "CommBackend", CommBackend)
    setattr(mnnvl_module, _MNNVL_PATCHED_MARKER, True)

    logger.info("patched flashinfer.comm.mnnvl.CommBackend placeholder")

