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

# CUDA normalization operator implementations using sgl_kernel.

from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_cuda(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    RMS normalization using sgl_kernel.

    Args:
        obj: The calling obj (provides obj.weight, obj.variance_epsilon)
        x: Input tensor
        residual: Optional residual tensor

    Returns:
        Normalized tensor, or tuple of (normalized, residual) if residual provided
    """
    from sgl_kernel import rmsnorm as sgl_rms_norm
    from sgl_kernel import fused_add_rmsnorm as sgl_fused_add_rms_norm

    weight = obj.weight
    epsilon = obj.variance_epsilon

    if residual is not None:
        sgl_fused_add_rms_norm(x, residual, weight, epsilon)
        return x, residual
    else:
        out = torch.empty_like(x)
        sgl_rms_norm(out, x, weight, epsilon)
        return out
