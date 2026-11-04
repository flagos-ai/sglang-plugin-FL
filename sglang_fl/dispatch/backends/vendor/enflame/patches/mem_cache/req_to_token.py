"""Use int32 GCU tensor indices when writing the request-to-KV mapping."""

import torch


def write_req_to_token_gcu(original, self, indices, values):
    table = self.req_to_token
    if (
        table.device.type == "gcu"
        and table.ndim == 2
        and isinstance(indices, tuple)
        and len(indices) == 2
        and all(
            isinstance(index, torch.Tensor)
            and index.device == table.device
            and index.ndim == 1
            and index.dtype in (torch.int32, torch.int64)
            for index in indices
        )
        and max(table.shape) <= torch.iinfo(torch.int32).max
    ):
        indices = tuple(index.to(torch.int32) for index in indices)
    return original(self, indices, values)


def patch_req_to_token():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.mem_cache.memory_pool.ReqToTokenPool.write",
        write_req_to_token_gcu,
        HookType.AROUND,
    )
