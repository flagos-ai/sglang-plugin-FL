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

    Delegates to SGLang's native ``UnquantizedFusedMoEMethod.forward_cuda``, which
    selects the runner backend (triton / deep_gemm / ...) and builds
    ``TritonMoeQuantInfo`` correctly via ``get_triton_quant_info``. This mirrors the
    MUSA backend (``obj.forward_musa``) and avoids re-implementing the quant_info
    contract, which drifts across sglang versions: sglang 0.5.12's runner reads
    ``use_fp8``/``use_int8``/... that a hand-built TritonMoeQuantInfo omits, raising
    AttributeError on the MoE forward.

    Args:
        obj: The UnquantizedFusedMoEMethod instance
        layer: The MoE layer module
        dispatch_output: StandardDispatchOutput containing hidden_states and topk_output

    Returns:
        CombineInput (StandardCombineInput)
    """
    return obj.forward_cuda(layer, dispatch_output)
