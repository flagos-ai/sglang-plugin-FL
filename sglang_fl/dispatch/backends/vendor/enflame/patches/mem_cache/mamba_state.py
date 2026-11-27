"""Keep Mamba slot clearing on the portable per-tensor path where the fused kernel cannot run.

``MambaPool.clear_slots`` / ``copy_from`` use the fused Triton slot kernels when
``_should_fuse_slot_ops()`` is true.  Those kernels pass pointer, stride and slot
tables as int64 tensors.  GCU400 accepts int64 tensor arguments; GCU300 rejects
them in the compiler and, with the check disabled, traps the device at run time.
Route GCU300 through sglang's portable per-tensor path instead (the behavior of
``SGLANG_DISABLE_FUSED_MAMBA_SLOT_OPS=1``).  Slot clears run once per newly
allocated request.
"""

import torch

_ARCH_MAJOR_CACHE = {}


def _fused_slot_ops_supported(device) -> bool:
    """Whether this device accepts the int64 tables the fused kernels build."""
    index = device.index
    if index is None:
        index = torch.gcu.current_device()

    supported = _ARCH_MAJOR_CACHE.get(index)
    if supported is None:
        major = torch.gcu.get_device_properties(index).major
        supported = major >= 4
        _ARCH_MAJOR_CACHE[index] = supported
    return supported


def should_fuse_slot_ops_gcu(original, self) -> bool:
    convs = self.mamba_cache.conv
    if convs and not _fused_slot_ops_supported(convs[0].device):
        return False
    return original(self)


# ---------------------------------------------------------------------------
# Patch registration
# ---------------------------------------------------------------------------

def patch_mamba_state():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.mem_cache.memory_pool.MambaPool._should_fuse_slot_ops",
        should_fuse_slot_ops_gcu,
        HookType.AROUND,
    )
