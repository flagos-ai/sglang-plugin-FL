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

"""Preserve eager positions and output ownership in the Ascend ViT graph path."""

from __future__ import annotations

from functools import wraps
import inspect

import torch


_PATCH_MARKER = "_sglang_fl_npu_vit_graph_parity"
_PREPARE_PARAMETERS = ("self", "x", "grid_thw")
_REPLAY_PARAMETERS = (
    "self",
    "graph_key",
    "x_3d",
    "rotary_pos_emb_cos",
    "rotary_pos_emb_sin",
    "output_indices",
)


def patch_qwen3_vl_npu_graph() -> None:
    """Keep ViT graph execution equivalent to eager and safe across replays.

    The pinned SGLang eager path uses linspace position interpolation, while
    graph preprocessing uses the legacy half-pixel path by default. Reuse the
    eager method and its vectorization decision only on NPU; leave CUDA graph
    preprocessing and ordinary eager forward unchanged.

    A replay returns the runner's reusable static output. Multimodal consumers
    can retain views while processing more images, so each invocation must own
    its output before a later replay overwrites that storage. Clone on the
    current device stream without introducing a host synchronization or an
    eager fallback.
    """
    from sglang.srt.hardware_backend.npu.graph_runner.vit_npu_graph_runner import (
        ViTNpuGraphRunner,
    )
    from sglang.srt.models import qwen3_vl

    vision_class = qwen3_vl.Qwen3VLMoeVisionModel
    original_prepare = vision_class._prepare_graph_inputs
    original_replay = ViTNpuGraphRunner.replay
    prepare_patched = getattr(original_prepare, _PATCH_MARKER, False)
    replay_patched = getattr(original_replay, _PATCH_MARKER, False)
    if prepare_patched and replay_patched:
        return
    if prepare_patched or replay_patched:
        raise RuntimeError("Incomplete Ascend ViT graph parity patch")

    for original, expected in (
        (original_prepare, _PREPARE_PARAMETERS),
        (original_replay, _REPLAY_PARAMETERS),
    ):
        actual = tuple(inspect.signature(original).parameters)
        if actual != expected:
            raise RuntimeError(
                "Unsupported SGLang interface for the Ascend ViT graph patch: "
                f"{original.__qualname__}{actual}, expected {expected}."
            )

    @wraps(original_prepare)
    def prepare(self, x, grid_thw):
        if self.device.type != "npu":
            return original_prepare(self, x, grid_thw)

        x = self.patch_embed(
            x.to(device=self.device, dtype=self.dtype, non_blocking=True)
        )
        if isinstance(grid_thw, list):
            grid_list = grid_thw
            grid_tensor = torch.tensor(grid_thw, dtype=torch.int32)
        else:
            grid_list = grid_thw.tolist()
            grid_tensor = grid_thw

        if self._use_vectorized_pos_embed(len(grid_list)):
            pos_embeds = self.fast_pos_embed_interpolate_vectorized(grid_list)
        else:
            pos_embeds = self.fast_pos_embed_interpolate_from_list(grid_list)
        x += pos_embeds
        cos, sin = self.rot_pos_emb(grid_list)
        cu_seqlens = qwen3_vl.compute_cu_seqlens_from_grid_numpy(grid_tensor)
        return x, cu_seqlens, cos, sin

    @wraps(original_replay)
    def replay(
        self,
        graph_key,
        x_3d,
        rotary_pos_emb_cos=None,
        rotary_pos_emb_sin=None,
        output_indices=None,
    ):
        output = original_replay(
            self,
            graph_key,
            x_3d,
            rotary_pos_emb_cos,
            rotary_pos_emb_sin,
            output_indices,
        )
        return output.clone()

    setattr(prepare, _PATCH_MARKER, True)
    setattr(replay, _PATCH_MARKER, True)
    vision_class._prepare_graph_inputs = prepare
    ViTNpuGraphRunner.replay = replay
