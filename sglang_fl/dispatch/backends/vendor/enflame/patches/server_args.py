def server_args_post_init(original_fn, self):
    self.device = "gcu"
    self.attention_backend = "fa3"
    self.mm_attention_backend = "fa3"
    self.page_size = 64
    self.watchdog_timeout = 100000
    self.disable_radix_cache = True
    original_fn(self)

def patch_server_args():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    server_args_class = "sglang.srt.server_args.ServerArgs"
    HookRegistry.register(f"{server_args_class}.__post_init__", server_args_post_init, HookType.AROUND)