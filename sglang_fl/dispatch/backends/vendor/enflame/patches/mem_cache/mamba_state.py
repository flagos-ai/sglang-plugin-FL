"""GCU Mamba slot clearing without advanced-index scatter kernels."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _zero_slots_kernel(
    state, indices,
    NINDICES: tl.constexpr, INDEX_STRIDE: tl.constexpr,
    NSLOTS: tl.constexpr, LAYERS: tl.constexpr,
    LAYER_STRIDE: tl.constexpr, SLOT_STRIDE: tl.constexpr,
    OUTER_SHAPE: tl.constexpr, OUTER_STRIDES: tl.constexpr,
    OUTER_DIVISORS: tl.constexpr, OUTER_SIZE: tl.constexpr,
    INNER_SIZE: tl.constexpr, INNER_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for pair in range(tl.program_id(0), LAYERS * NINDICES * OUTER_SIZE,
                      tl.num_programs(0)):
        layer = pair // (NINDICES * OUTER_SIZE)
        index_id = (pair // OUTER_SIZE) % NINDICES
        outer = pair % OUTER_SIZE
        slot = tl.load(indices + index_id * INDEX_STRIDE).to(tl.int64)
        slot = tl.where(slot < 0, slot + NSLOTS, slot)
        base = state + layer * LAYER_STRIDE + slot * SLOT_STRIDE
        for axis in tl.static_range(len(OUTER_SHAPE)):
            coordinate = (outer // OUTER_DIVISORS[axis]) % OUTER_SHAPE[axis]
            base = base + coordinate * OUTER_STRIDES[axis]
        for block in range(tl.program_id(1), tl.cdiv(INNER_SIZE, BLOCK),
                           tl.num_programs(1)):
            offsets = block * BLOCK + tl.arange(0, BLOCK)
            tl.store(base + offsets * INNER_STRIDE, 0, offsets < INNER_SIZE)


def zero_selected_slots_gcu(state, indices):
    """Clear valid dim-1 slots on device without touching unselected slots."""
    if state.numel() == 0 or indices.numel() == 0:
        return
    if state.device.type != "gcu" or indices.device != state.device:
        raise ValueError("GCU state and indices must be on the same device")
    if state.ndim < 2 or indices.ndim != 1:
        raise ValueError("Expected [layer, slot, ...] state and 1-D indices")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("Slot indices must have int32 or int64 dtype")
    # Merge physically contiguous tail dimensions, including stepped views.
    shape = list(state.shape[2:])
    strides = list(state.stride()[2:])
    if not shape:
        shape, strides = [1], [1]
    while len(shape) > 1 and strides[-2] == strides[-1] * shape[-1]:
        shape[-2] *= shape[-1]
        strides[-2] = strides[-1]
        shape.pop()
        strides.pop()
    outer_shape = tuple(shape[:-1])
    outer_strides = tuple(strides[:-1])
    divisors = tuple(
        math.prod(outer_shape[axis + 1:]) for axis in range(len(outer_shape))
    )
    outer_size = math.prod(outer_shape)
    _zero_slots_kernel[
        (min(state.shape[0] * indices.numel() * outer_size, 65535),
         min(triton.cdiv(shape[-1], 128), 255))
    ](
        state, indices, indices.numel(), indices.stride(0),
        state.shape[1], state.shape[0], state.stride(0), state.stride(1),
        outer_shape, outer_strides, divisors, outer_size,
        shape[-1], strides[-1], 128,
    )


def clear_slots_gcu(original, self, indices):
    convs = self.mamba_cache.conv
    if (
        not convs
        or convs[0].device.type != "gcu"
        or not self._should_fuse_slot_ops()
    ):
        return original(self, indices)

    for state in convs:
        zero_selected_slots_gcu(state, indices)
    zero_selected_slots_gcu(self.mamba_cache.temporal, indices)


def patch_mamba_state():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.mem_cache.memory_pool.MambaPool.clear_slots",
        clear_slots_gcu,
        HookType.AROUND,
    )
