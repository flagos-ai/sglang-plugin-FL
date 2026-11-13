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

# Ascend mhc_sinkhorn operator implementation.
#
# On Ascend the fused Triton kernel is the same FlagOS implementation: the
# FlagTree Triton backend compiles it for NPU. Kept as a thin vendor layer
# so future Ascend-native (CANN custom op) versions can slot in without
# changing the dispatch contract.

from __future__ import annotations

import torch


def mhc_sinkhorn_ascend(comb: torch.Tensor, eps: float, repeat: int) -> torch.Tensor:
    """Apply fused Sinkhorn normalization; Ascend entry of mhc_sinkhorn.

    Delegates to the FlagOS fused kernel (Triton-Ascend). See
    sglang_fl.dispatch.backends.flagos.impl.mhc for the implementation and
    the graph-mode performance note.

    Args:
        comb: [.., N, N] contiguous tensor, modified in place.
        eps: additive epsilon, matching the reference numerics.
        repeat: total number of normalization rounds (>= 1).

    Returns:
        The same tensor object with normalized values.
    """
    from sglang_fl.dispatch.backends.flagos.impl.mhc import mhc_sinkhorn_flagos

    return mhc_sinkhorn_flagos(comb, eps, repeat)
