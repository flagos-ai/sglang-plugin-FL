def patch_mem_cache():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from sglang_fl.dispatch.backends.vendor.enflame.patches.mem_cache.common import write_cache_indices
    HookRegistry.register(f"sglang.srt.mem_cache.common.write_cache_indices", write_cache_indices, HookType.REPLACE)