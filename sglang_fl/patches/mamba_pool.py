# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Chunk ``MambaPool.copy_from`` to bound its transient allocation.

``temporal[:, dst] = temporal[:, src]`` gathers every source slot into one
temporary of ``n_slots * slot_bytes`` before scattering. Under an admission
burst of COW copies this is several GiB and OOMs the server (issue #209).
Upstream fixed it in sgl-project/sglang#47984 (commit 784cffe3d9), which no
v0.5.x release carries.

Upstream chunks only the temporal copy inside ``copy_from``. Here the whole
call is split into slot chunks and the original ``copy_from`` runs once per
chunk, so the patch does not copy any version-specific function body. Every
step in ``copy_from`` (conv copy, temporal copy, ReplaySSM cursor reset) is
per-slot, so for disjoint src/dst slots the result is identical. All SGLang
callers satisfy that: COW copies a radix checkpoint slot into a freshly
allocated active slot, and ``UnifiedMambaPool._copy_from_physical`` moves
survivors from the vacated band into holes above the new watermark.
"""

_TARGET = "sglang.srt.mem_cache.memory_pool.MambaPool.copy_from"

# Same value as upstream #47984 (_MAMBA_TEMPORAL_COPY_CHUNK_BYTES).
_CHUNK_BYTES = 64 * 1024 * 1024


def chunked_copy_from(original, self, src_indices, dst_indices):
    # Base SGLang already chunks the temporal copy (#47984 is in).
    if hasattr(self, "_copy_temporal_slots"):
        return original(self, src_indices, dst_indices)

    temporal = self.mamba_cache.temporal
    if temporal.numel() == 0 or dst_indices.dim() != 1:
        return original(self, src_indices, dst_indices)

    num_slots = dst_indices.shape[0]
    slot_bytes = temporal.element_size() * (temporal.numel() // temporal.shape[1])
    step = max(1, _CHUNK_BYTES // slot_bytes)
    if num_slots <= step:
        return original(self, src_indices, dst_indices)

    if getattr(self, "debug_memory_pool", False):
        overlap = set(src_indices.tolist()) & set(dst_indices.tolist())
        assert not overlap, (
            "chunked copy_from requires disjoint src/dst slots; "
            f"overlap={sorted(overlap)}"
        )
    for start in range(0, num_slots, step):
        chunk = slice(start, start + step)
        original(self, src_indices[chunk], dst_indices[chunk])


def patch_mamba_pool_copy_from():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(_TARGET, chunked_copy_from, HookType.AROUND)
