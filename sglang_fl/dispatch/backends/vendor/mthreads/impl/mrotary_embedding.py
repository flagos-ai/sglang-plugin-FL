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

# MUSA MRotaryEmbedding operator implementation.
#
# TODO: Replace NotImplementedError with torch_musa native kernel once verified
# on hardware. Current behavior: falls back to reference.

from __future__ import annotations

from typing import Tuple

import torch


def mrotary_embedding_musa(
    obj,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-modal rotary position embedding on MUSA.

    Args:
        obj: The MRotaryEmbedding instance
        positions: Position tensor (1D or 2D for multi-modal)
        query: Query tensor
        key: Key tensor

    Returns:
        Tuple of (embedded_query, embedded_key)
    """
    raise NotImplementedError(
        "mrotary_embedding_musa: no torch_musa kernel wired yet; falling back to flaggems/reference"
    )
