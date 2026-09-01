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

# Ascend normalization operator implementations.
# rms_norm      : torch_npu.npu_rms_norm / npu_add_rms_norm (fused add+norm)
# gemma_rms_norm: torch_npu.npu_gemma_rms_norm / sgl_kernel_npu add_gemma_rms_norm

from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_ascend(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    RMS normalization using Ascend NPU.

    Args:
        obj: The calling obj (provides obj.weight, obj.variance_epsilon)
        x: Input tensor
        residual: Optional residual tensor

    Returns:
        Normalized tensor, or tuple of (normalized, residual) if residual provided
    """
    import torch_npu

    weight = obj.weight
    epsilon = obj.variance_epsilon

    if residual is not None:
        x, _, residual = torch_npu.npu_add_rms_norm(x, residual, weight, epsilon)
        return x, residual

    x, _ = torch_npu.npu_rms_norm(x, weight, epsilon)
    return x


def gemma_rms_norm_ascend(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    
    import torch_npu
    from sgl_kernel_npu.norm.add_rmsnorm_bias import add_gemma_rms_norm

    weight = obj.weight
    epsilon = obj.variance_epsilon

    if residual is not None:
        norm_out, residual = add_gemma_rms_norm(x, weight, residual, epsilon)
        return norm_out, residual

    x, _ = torch_npu.npu_gemma_rms_norm(x, weight, epsilon)
    return x
