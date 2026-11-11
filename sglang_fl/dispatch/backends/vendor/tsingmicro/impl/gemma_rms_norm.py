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

"""GemmaRMSNorm on TXDA, backed by FlagGems' ``_tsingmicro`` kernels.

SGLang's ``GemmaRMSNorm`` (the dense norm in Qwen3.5/3.6) is plain RMSNorm
with the weight shifted by one::

    out = (x / sqrt(mean(x^2) + eps)) * (1 + weight)

``sglang.srt.layers.layernorm.GemmaRMSNorm.forward_native`` upcasts to fp32
before applying ``1 + weight``, so the offset must be built in fp32 as well
(``1.0 + weight.float()``) -- adding it in bf16 loses mantissa bits and biases
every normalized element by up to ~0.4%.

FlagGems ships ``flag_gems.ops.gemma_rms_norm`` only as a bare Triton kernel
(no Python wrapper), so this implementation drives the standard RMSNorm entry
points with the precomputed fp32 weight::

    flag_gems.rms_norm(x, [N], gemma_weight, eps)                   # no residual
    flag_gems.fused_add_rms_norm(x, residual, [N], gemma_weight, eps)  # fused add

Both names are rebound by FlagGems' ``SpecOpRegistrar`` to the ``_tsingmicro``
vendor implementations at import time (``flag_gems.rms_norm`` ->
``_tsingmicro/ops/rms_norm.py``, ``flag_gems.fused_add_rms_norm`` ->
``_tsingmicro/fused/fused_add_rms_norm.py``), so this reaches the tuned TXDA
multi-row kernels rather than the generic ones or the PyTorch reference.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

# Cache slot names on the calling GemmaRMSNorm module.
_CACHE_ATTR = "_fl_gemma_weight_fp32"
_CACHE_SRC_ATTR = "_fl_gemma_weight_src"


def _gemma_weight_fp32(obj) -> torch.Tensor:
    """Return ``1.0 + weight.float()`` for the calling norm module.

    Computed once per loaded weight instead of once per token: FlagGems'
    kernels accept an fp32 weight alongside bf16 activations, so this keeps the
    exact ``forward_native`` numerics without a per-call cast/allocation.

    sglang maintains ``obj.gemma_weight`` and reassigns it in
    ``GemmaRMSNorm._weight_loader`` whenever weights are (re)loaded, so its
    object identity is used as the cache generation marker.  On sglang builds
    without that attribute the weight is rebuilt on every call, which is always
    correct but slightly slower.
    """
    src = getattr(obj, "gemma_weight", None)
    cached = getattr(obj, _CACHE_ATTR, None)
    if (
        src is not None
        and cached is not None
        and getattr(obj, _CACHE_SRC_ATTR, None) is src
    ):
        return cached

    weight = (obj.weight.float() + 1.0).contiguous()
    if src is not None:
        obj.__dict__[_CACHE_ATTR] = weight
        obj.__dict__[_CACHE_SRC_ATTR] = src
    return weight


def gemma_rms_norm_txda(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """GemmaRMSNorm via FlagGems RMSNorm kernels.

    Mirrors ``GemmaRMSNorm.forward_native``:

    - ``residual is None``  -> the normalized tensor
    - ``residual is not None`` -> ``(normalized, x + residual)``, where the
      returned residual is the pre-norm sum (FlagGems' fused add writes it
      back in place).
    """
    import flag_gems

    weight = _gemma_weight_fp32(obj)
    normalized_shape = [weight.numel()]
    eps = obj.variance_epsilon

    if residual is None:
        return flag_gems.rms_norm(x, normalized_shape, weight, eps)
    return flag_gems.fused_add_rms_norm(x, residual, normalized_shape, weight, eps)
