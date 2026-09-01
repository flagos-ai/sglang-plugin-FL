
def patch_communicator_hooks():
    from sglang.srt.distributed.parallel_state import _DEVICE_TO_DISTRIBUTED_BACKEND as db
    db["gcu"] = "eccl"

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from sglang_fl.dispatch.backends.vendor.enflame.patches.utils.common import is_cuda_alike
    HookRegistry.register(f"sglang.srt.utils.common.is_cuda_alike", is_cuda_alike, HookType.REPLACE)

    from sglang_fl.dispatch.backends.vendor.enflame.patches.device_communicators.pynccl_wrapper import NCCLLibrary_init
    from sglang_fl.dispatch.backends.vendor.enflame.patches.device_communicators.cuda_wrapper import CudaRTLibrary_init
    HookRegistry.register(f"sglang.srt.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.__init__", NCCLLibrary_init, HookType.REPLACE)
    HookRegistry.register(f"sglang.srt.distributed.device_communicators.cuda_wrapper.CudaRTLibrary.__init__", CudaRTLibrary_init, HookType.REPLACE)
