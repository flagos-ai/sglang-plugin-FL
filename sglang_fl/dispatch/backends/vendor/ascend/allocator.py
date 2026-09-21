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

"""Ascend-compatible paged KV-cache allocator for SGLang 0.5.18."""

from __future__ import annotations

import torch

from sglang.srt.hardware_backend.npu.allocator_npu import (
    NPUPagedTokenToKVPoolAllocator,
)

# torch_npu's arange implementation on the CANN 8.5 / torch_npu 2.8 stack uses
# one AICore block per element and rejects a one-dimensional launch larger than
# 65,535 blocks. A 910C KV cache routinely contains hundreds of thousands of
# pages, so SGLang's single arange in PagedTokenToKVPoolAllocator.clear() is too
# large. Keep each launch within the device limit and concatenate the identical
# [1, num_pages] sequence.
ASCEND_ARANGE_MAX_ELEMENTS = 65_535


def build_ascend_page_range(
    num_pages: int,
    *,
    device: str | torch.device,
    max_elements: int = ASCEND_ARANGE_MAX_ELEMENTS,
) -> torch.Tensor:
    """Return page ids ``[1, num_pages]`` using bounded arange launches."""

    if num_pages < 0:
        raise ValueError(f"num_pages must be non-negative, got {num_pages}")
    if max_elements <= 0:
        raise ValueError(f"max_elements must be positive, got {max_elements}")
    if num_pages == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)

    stop = num_pages + 1
    chunks = [
        torch.arange(
            start,
            min(start + max_elements, stop),
            dtype=torch.int64,
            device=device,
        )
        for start in range(1, stop, max_elements)
    ]
    return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class AscendPagedTokenToKVPoolAllocator(NPUPagedTokenToKVPoolAllocator):
    """NPU allocator with a CANN-safe free-page reset."""

    def clear(self) -> None:
        self.free_pages = build_ascend_page_range(
            self.num_pages,
            device=self.device,
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.free_page_reps_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)
