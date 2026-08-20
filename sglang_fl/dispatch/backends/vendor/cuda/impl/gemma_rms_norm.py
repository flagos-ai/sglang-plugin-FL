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

# CUDA vendor GemmaRMSNorm — delegates to SGLang's native sgl_kernel.

from __future__ import annotations

from typing import Optional, Union

import torch


def gemma_rms_norm_cuda(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    GemmaRMSNorm using SGLang's native CUDA kernel (sgl_kernel).

    Delegates to obj._forward_impl which calls sgl_kernel.gemma_rmsnorm.
    """
    return obj._forward_impl(x, residual)
