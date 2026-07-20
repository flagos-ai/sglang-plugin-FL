from typing import Optional
import ctypes
import re
from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary,find_loaded_library

def CudaRTLibrary_init(self, so_file: Optional[str] = None):
    if so_file is None:
        so_file = find_loaded_library("libtopsrt") #FOR_GCU
        assert so_file is not None, "libtopsrt is not loaded in the current process"
    if so_file not in CudaRTLibrary.path_to_library_cache:
        lib = ctypes.CDLL(so_file)
        CudaRTLibrary.path_to_library_cache[so_file] = lib
    self.lib = CudaRTLibrary.path_to_library_cache[so_file]

    if so_file not in CudaRTLibrary.path_to_dict_mapping:
        _funcs = {}
        for func in CudaRTLibrary.exported_functions:
            f = getattr(self.lib, re.sub(r"^cuda", "tops", func.name)) # FOR_GCU
            f.restype = func.restype
            f.argtypes = func.argtypes
            _funcs[func.name] = f
        CudaRTLibrary.path_to_dict_mapping[so_file] = _funcs
    self.funcs = CudaRTLibrary.path_to_dict_mapping[so_file]