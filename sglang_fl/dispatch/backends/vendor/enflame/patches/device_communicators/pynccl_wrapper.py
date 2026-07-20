from typing import Optional,Dict,Any
import re
import ctypes
import platform
from sglang.srt.distributed.device_communicators.pynccl_wrapper import NCCLLibrary, logger


def NCCLLibrary_init(self, so_file: Optional[str] = None):

    so_file = "libeccl.so" # FOR_GCU

    try:
        if so_file not in NCCLLibrary.path_to_dict_mapping:
            lib = ctypes.CDLL(so_file)
            NCCLLibrary.path_to_library_cache[so_file] = lib
        self.lib = NCCLLibrary.path_to_library_cache[so_file]
    except Exception as e:
        logger.error(
            "Failed to load NCCL library from %s . "
            "It is expected if you are not running on NVIDIA/AMD/MTHREADS GPUs. "
            "Otherwise, the nccl library might not exist, be corrupted "
            "or it does not support the current platform %s. "
            "If you already have the library, please set the "
            "environment variable SGLANG_NCCL_SO_PATH"
            " to point to the correct nccl library path.",
            so_file,
            platform.platform(),
        )
        raise e

    if so_file not in NCCLLibrary.path_to_dict_mapping:
        _funcs: Dict[str, Any] = {}
        exported_functions = NCCLLibrary.exported_functions
        if hasattr(self.lib, "ncclCommWindowRegister"):
            exported_functions.extend(NCCLLibrary.exported_functions_symm_mem)
        for func in exported_functions:
            f = getattr(self.lib, re.sub(r"^nccl", "eccl", func.name)) # FOR_GCU
            f.restype = func.restype
            f.argtypes = func.argtypes
            _funcs[func.name] = f
        NCCLLibrary.path_to_dict_mapping[so_file] = _funcs
    self._funcs = NCCLLibrary.path_to_dict_mapping[so_file]