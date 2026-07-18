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

# Bridge: RMSNorm
#
# SGLang signature:
#   forward_cuda(self, x, residual=None, post_residual_addition=None)
#     -> Tensor | tuple[Tensor, Tensor]
#
# Dispatch signature:
#   fn(obj, x, residual=None) -> Tensor | tuple[Tensor, Tensor]
#
# SGLang-specific handling:
#   - post_residual_addition: added to residual before passing to dispatch

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from sglang_fl.dispatch import call_op


def rms_norm_bridge(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    post_residual_addition: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """SGLang RMSNorm forward → dispatch call_op("rms_norm", ...).

    Handles SGLang-specific parameters before delegating to dispatch.
    """
    # Handle post_residual_addition: merge into residual
    if post_residual_addition is not None and residual is not None:
        residual = residual + post_residual_addition
    elif post_residual_addition is not None and residual is None:
        residual = post_residual_addition

    return call_op("rms_norm", self, x, residual)
