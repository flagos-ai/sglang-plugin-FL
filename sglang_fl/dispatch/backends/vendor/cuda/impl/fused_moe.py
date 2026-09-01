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

# CUDA FusedMoE operator implementation using SGLang's native fused_experts.

from __future__ import annotations

import torch


def fused_moe_cuda(
    obj,
    layer: torch.nn.Module,
    dispatch_output,
):
    """
    Fused MoE expert computation using SGLang's native Triton kernels.

    This implementation delegates to SGLang's MoeRunner with the TRITON backend,
    which uses the standard (non-triton_kernels, non-flashinfer) path.

    Args:
        obj: The UnquantizedFusedMoEMethod instance
        layer: The MoE layer module
        dispatch_output: StandardDispatchOutput containing hidden_states and topk_output

    Returns:
        CombineInput (StandardCombineInput)
    """
    from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo

    # Use the TRITON backend (standard path, not triton_kernels or flashinfer)
    # This matches the final fallback in forward_cuda
    quant_info = TritonMoeQuantInfo(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        b13=getattr(layer, "w13_weight_bias", None),
        b2=getattr(layer, "w2_weight_bias", None),
    )
    return obj.runner.run(dispatch_output, quant_info)
