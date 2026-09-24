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

"""SGLang 0.5.18 Ascend normalization adapters."""

from __future__ import annotations

from typing import Optional, Union

import torch


def _forward_npu(obj, x: torch.Tensor, residual: Optional[torch.Tensor]):
    forward_npu = getattr(obj, "forward_npu", None)
    if forward_npu is None:
        raise RuntimeError(
            f"SGLang 0.5.18 {type(obj).__name__}.forward_npu is required by "
            "the Ascend normalization adapter"
        )
    return forward_npu(x, residual)


def rms_norm_ascend(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Use SGLang's native NPU RMSNorm contract."""

    return _forward_npu(obj, x, residual)


def gemma_rms_norm_ascend(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Use SGLang's native NPU Gemma RMSNorm and its size/env fallbacks."""

    return _forward_npu(obj, x, residual)
