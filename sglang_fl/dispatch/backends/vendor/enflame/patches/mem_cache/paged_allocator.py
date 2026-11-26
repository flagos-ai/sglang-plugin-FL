"""Use int32 paged KV slot indices when allocating on GCU."""

import torch


def _index_dtype(allocator):
    """Return int32 when the slot range fits, None to keep the original path."""
    if allocator.num_pages > torch.iinfo(torch.int32).max // allocator.page_size:
        return None
    return torch.int32


def alloc_extend_gcu(
    original,
    self,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
    num_new_pages: int = None,
):
    dtype = _index_dtype(self)
    if dtype is None:
        return original(
            self,
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            num_new_pages,
        )

    free_pages = self.free_pages
    self.free_pages = free_pages.to(dtype)
    try:
        out_indices = original(
            self,
            prefix_lens.to(dtype),
            prefix_lens_cpu,
            seq_lens.to(dtype),
            seq_lens_cpu,
            last_loc.to(dtype),
            extend_num_tokens,
            num_new_pages,
        )
    finally:
        self.free_pages = self.free_pages.to(free_pages.dtype)

    if out_indices is None:
        return None
    return out_indices.to(torch.int64)


def alloc_decode_gcu(original, self, seq_lens, seq_lens_cpu, last_loc):
    dtype = _index_dtype(self)
    if dtype is None:
        return original(self, seq_lens, seq_lens_cpu, last_loc)

    free_pages = self.free_pages
    self.free_pages = free_pages.to(dtype)
    try:
        out_indices = original(
            self,
            seq_lens.to(dtype),
            seq_lens_cpu,
            last_loc.to(dtype),
        )
    finally:
        self.free_pages = self.free_pages.to(free_pages.dtype)

    if out_indices is None:
        return None
    return out_indices.to(torch.int64)


# ---------------------------------------------------------------------------
# Patch registration
# ---------------------------------------------------------------------------

def patch_paged_allocator():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.mem_cache.allocator.paged.PagedTokenToKVPoolAllocator.alloc_extend",
        alloc_extend_gcu,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.mem_cache.allocator.paged.PagedTokenToKVPoolAllocator.alloc_decode",
        alloc_decode_gcu,
        HookType.AROUND,
    )
