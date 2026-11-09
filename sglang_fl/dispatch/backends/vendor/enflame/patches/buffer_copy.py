"""Use individual tensor copies for GCU graph and eager input buffers."""


def grouped_copy_gcu(original, dsts, srcs):
    if not any(t.device.type == "gcu" for t in (*dsts, *srcs)):
        return original(dsts, srcs)
    for dst, src in zip(dsts, srcs):
        dst.copy_(src)


def patch_buffer_copy():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.model_executor.cuda_graph_buffer_registry._grouped_foreach_copy_",
        grouped_copy_gcu,
        HookType.AROUND,
    )

    HookRegistry.register(
        "sglang.srt.model_executor.runner_utils.buffers._grouped_foreach_copy_",
        grouped_copy_gcu,
        HookType.AROUND,
    )
