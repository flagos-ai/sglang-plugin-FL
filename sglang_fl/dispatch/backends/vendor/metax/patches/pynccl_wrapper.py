"""Point SGLang's pynccl wrapper at MetaX MCCL.

Based on ``NCCLLibrary.__init__`` in
``sglang/srt/distributed/device_communicators/pynccl_wrapper.py`` (SGLang
v0.5.18, Apache-2.0). MCCL exports the NCCL C API from ``libmccl.so`` under an
``mccl`` prefix (``mcclAllReduce`` ...). Only the library name and the symbol
lookup differ; the function table, argument types and methods stay upstream.
"""

from __future__ import annotations

import ctypes
import importlib
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_MODULE = "sglang.srt.distributed.device_communicators.pynccl_wrapper"
_MCCL_LIBRARY = "libmccl.so"
_PATCHED_MARKER = "__sglang_fl_metax_mccl__"


def mccl_symbol(name: str) -> str:
    """Map an NCCL API name to its MCCL export (ncclAllReduce -> mcclAllReduce)."""
    return "m" + name[1:] if name.startswith("nccl") else name


def patch_pynccl_wrapper() -> None:
    """Make ``NCCLLibrary`` load ``libmccl.so`` and bind the ``mccl*`` symbols.

    The class is patched in place, so modules that already imported
    ``NCCLLibrary`` see the change. This only runs from the MetaX patch set.
    """
    module = importlib.import_module(_MODULE)
    nccl_library = module.NCCLLibrary
    if getattr(nccl_library.__init__, _PATCHED_MARKER, False):
        return

    def __init__(self, so_file: Optional[str] = None):
        # Upstream would pick libnccl.so.2 because the MetaX torch reports
        # torch.version.cuda. SGLANG_NCCL_SO_PATH still takes precedence.
        so_file = so_file or os.environ.get("SGLANG_NCCL_SO_PATH") or _MCCL_LIBRARY
        if so_file not in nccl_library.path_to_dict_mapping:
            lib = ctypes.CDLL(so_file)
            functions = list(nccl_library.exported_functions)
            if hasattr(lib, mccl_symbol("ncclCommWindowRegister")):
                functions += nccl_library.exported_functions_symm_mem
            funcs = {}
            for func in functions:
                f = getattr(lib, mccl_symbol(func.name))
                f.restype = func.restype
                f.argtypes = func.argtypes
                funcs[func.name] = f
            nccl_library.path_to_library_cache[so_file] = lib
            nccl_library.path_to_dict_mapping[so_file] = funcs
        self.lib = nccl_library.path_to_library_cache[so_file]
        self._funcs = nccl_library.path_to_dict_mapping[so_file]

    setattr(__init__, _PATCHED_MARKER, True)
    nccl_library.__init__ = __init__
    logger.info("patched NCCLLibrary to load %s (mccl* symbols)", _MCCL_LIBRARY)
