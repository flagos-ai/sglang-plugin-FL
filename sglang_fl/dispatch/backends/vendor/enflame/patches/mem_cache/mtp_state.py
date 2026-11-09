"""GCU copies for speculative KV and recurrent state commits."""

from __future__ import annotations

import torch


def init_kv_copy_gcu(original, self):
    if torch.device(self.device).type != "gcu":
        return original(self)

    from sglang.srt.mem_cache.memory_pool import move_kv_cache_native

    self.use_native_move_kv_cache = True
    self._kv_copy_config = None
    if self.layer_num == 0:
        return
    torch.gcu.synchronize()
    dummy = torch.zeros(1, dtype=torch.int64, device=self.device)
    move_kv_cache_native(self.k_buffer, self.v_buffer, dummy, dummy)
    if getattr(self, "k_scale_buffer", None) is not None:
        move_kv_cache_native(self.k_scale_buffer, self.v_scale_buffer, dummy, dummy)
    torch.gcu.synchronize()


def _copy_states(dst, src, slots, steps):
    """Copy selected windows, including noncontiguous conv-window views."""
    if len(slots) != len(steps):
        raise ValueError("Mamba destination and step index lengths differ")
    if dst.shape[0] != src.shape[0] or dst.shape[2:] != src.shape[3:]:
        raise ValueError("Mamba source and destination state shapes differ")
    for req, (slot, step) in enumerate(zip(slots, steps)):
        if (
            step < 0
            or slot < 0
            or slot >= dst.shape[1]
            or req >= src.shape[1]
            or step >= src.shape[2]
        ):
            continue
        for layer in range(dst.shape[0]):
            dst[layer, slot].copy_(src[layer, req, step])


def scatter_mamba_gcu(
    original,
    mamba_caches,
    state_indices_tensor,
    last_correct_step_indices,
    mamba_track_indices,
    mamba_steps_to_track,
):
    if state_indices_tensor.device.type != "gcu":
        return original(
            mamba_caches,
            state_indices_tensor,
            last_correct_step_indices,
            mamba_track_indices,
            mamba_steps_to_track,
        )
    slots = state_indices_tensor.cpu().tolist()
    steps = last_correct_step_indices.cpu().tolist()
    track = None
    if mamba_track_indices is not None:
        if mamba_steps_to_track is None:
            raise ValueError("Mamba tracking requires track step indices")
        track = (
            mamba_track_indices.cpu().tolist(),
            mamba_steps_to_track.cpu().tolist(),
        )
    pairs = list(zip(mamba_caches.conv, mamba_caches.intermediate_conv_window))
    if mamba_caches.temporal.numel() > 0:
        pairs.insert(0, (mamba_caches.temporal, mamba_caches.intermediate_ssm))
    for dst, src in pairs:
        _copy_states(dst, src, slots, steps)
        if track is not None:
            _copy_states(dst, src, *track)


def patch_mtp_state():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType


    HookRegistry.register(
        "sglang.srt.mem_cache.memory_pool.MHATokenToKVPool._init_kv_copy_and_warmup",
        init_kv_copy_gcu,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.kernels.ops.mamba.mamba_state_scatter_triton.scatter_mamba_states_after_mtp_verify",
        scatter_mamba_gcu,
        HookType.AROUND,
    )
