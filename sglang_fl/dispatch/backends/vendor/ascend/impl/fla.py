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

# Ascend FLA (Flash Linear Attention) operator implementations.
#
# chunk_gated_delta_rule: sgl_kernel_npu.fla.chunk.chunk_gated_delta_rule_npu
# fused_recurrent_* : never called on NPU — AscendGDNAttnBackend routes around them.

from __future__ import annotations

from typing import Optional, Tuple

import torch


def chunk_gated_delta_rule_ascend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu

    # chunk_gated_delta_rule_npu does not accept initial_state_indices; pre-index here as sglang does.
    if initial_state is not None and initial_state_indices is not None:
        initial_state = initial_state[initial_state_indices]

    return chunk_gated_delta_rule_npu(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        head_first=head_first,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
