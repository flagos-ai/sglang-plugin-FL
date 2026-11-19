def get_available_gpu_memory_gcu(
    original, device, gpu_id, distributed=False, empty_cache=True, cpu_group=None
):
    import torch

    if device not in ("cuda", "gcu"):
        return original(device, gpu_id, distributed, empty_cache, cpu_group)
    if empty_cache:
        torch.gcu.empty_cache()
    free_memory, _ = torch.gcu.mem_get_info(gpu_id)
    if distributed:
        free_tensor = torch.tensor(float(free_memory), dtype=torch.float32, device="cpu")
        torch.distributed.all_reduce(
            free_tensor, op=torch.distributed.ReduceOp.MIN, group=cpu_group
        )
        free_memory = free_tensor.item()
    return free_memory / (1 << 30)


def patch_runtime_memory():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.utils.common.get_available_gpu_memory",
        get_available_gpu_memory_gcu,
        HookType.AROUND,
    )
