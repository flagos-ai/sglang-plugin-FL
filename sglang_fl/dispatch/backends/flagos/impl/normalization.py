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

# FlagOS normalization operator implementations.

from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_flagos(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    RMS normalization using FlagGems.

    Args:
        obj: The calling nn.Module (provides obj.weight, obj.variance_epsilon)
        x: Input tensor
        residual: Optional residual tensor

    Returns:
        Normalized tensor, or tuple of (normalized, residual) if residual provided
    """
    from flag_gems.modules.normalization import gems_rms_forward

    weight = obj.weight
    epsilon = obj.variance_epsilon

    return gems_rms_forward(x, residual, weight, epsilon)


def gemma_rms_norm_flagos(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Gemma-style RMS normalization using FlagGems: norm(x) * (1 + weight).

    Same semantics as sglang GemmaRMSNorm._forward_impl: with residual, both
    tensors are updated in place (residual = x + residual, x = norm(residual)).
    (1 + weight) is formed in fp32; bf16 cannot represent small offsets from 1.
    """
    import flag_gems
    from flag_gems.fused import fused_add_rms_norm

    hidden = x.shape[-1]
    weight = obj.weight.float() + 1.0
    epsilon = obj.variance_epsilon

    if residual is not None:
        return fused_add_rms_norm(x, residual, [hidden], weight, epsilon)

    out = flag_gems.rms_norm(x.contiguous(), [hidden], weight, epsilon)
    return out.view(x.shape)
