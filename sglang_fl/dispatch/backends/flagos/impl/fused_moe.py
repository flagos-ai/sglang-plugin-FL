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

"""FlagOS implementation of the unquantized SGLang fused-MoE bridge."""

from __future__ import annotations

import torch


def _validate_supported_config(obj, dispatch_output) -> None:
    """Reject configurations whose semantics FlagGems cannot preserve yet.

    Raising ``NotImplementedError`` is intentional: in fallback-enabled mode the
    FL dispatcher will try the next implementation instead of silently running
    a kernel with incompatible weights or routing semantics.
    """

    config = obj.moe_runner_config

    if getattr(obj, "use_triton_kernels", False):
        raise NotImplementedError(
            "FlagOS fused_moe expects the standard [E, N, K] weight layout"
        )

    if dispatch_output.hidden_states_scale is not None:
        raise NotImplementedError(
            "FlagOS fused_moe does not support pre-quantized dispatch input"
        )

    if config.activation != "silu" or not config.is_gated:
        raise NotImplementedError(
            "FlagOS fused_moe currently supports gated SiLU only"
        )

    if config.no_combine:
        raise NotImplementedError("FlagOS fused_moe does not support no_combine")

    if config.gemm1_alpha is not None or config.gemm1_clamp_limit is not None:
        raise NotImplementedError(
            "FlagOS fused_moe does not support custom SwiGLU alpha/clamp yet"
        )

    num_experts = config.num_experts
    num_local_experts = config.num_local_experts
    if (
        num_experts is None
        or num_local_experts is None
        or num_experts != num_local_experts
    ):
        raise NotImplementedError(
            "FlagOS fused_moe EP expert mapping is not supported yet"
        )


def fused_moe_flagos(
    obj,
    layer: torch.nn.Module,
    dispatch_output,
):
    """Run a complete unquantized MoE expert pass with FlagGems.

    SGLang has already performed routing, so this implementation consumes the
    materialized ``topk_weights`` and ``topk_ids`` and delegates expert
    alignment, both expert GEMMs, gated activation, and top-k reduction to
    ``flag_gems.fused.fused_experts_impl``.
    """

    _validate_supported_config(obj, dispatch_output)

    from flag_gems.fused import fused_experts_impl
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
    )

    config = obj.moe_runner_config
    topk_output = dispatch_output.topk_output
    topk_weights = topk_output.topk_weights
    topk_ids = topk_output.topk_ids

    # FlagGems' alignment kernels consume int32 expert indices. SGLang routing
    # commonly materializes them as int64.
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(dtype=torch.int32)

    output = fused_experts_impl(
        hidden_states=dispatch_output.hidden_states,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        # Start out-of-place to avoid aliasing SGLang dispatch buffers. This is
        # safer for fallback validation and preserves the combine contract.
        inplace=False,
        activation=config.activation,
        apply_router_weight_on_input=config.apply_router_weight_on_input,
        global_num_experts=config.num_experts,
        w1_bias=getattr(layer, "w13_weight_bias", None),
        w2_bias=getattr(layer, "w2_weight_bias", None),
    )

    # SGLang's Triton runner applies this factor after combining expert
    # outputs; it is not part of FlagGems' fused_experts_impl signature.
    routed_scaling_factor = config.routed_scaling_factor
    if routed_scaling_factor is not None and routed_scaling_factor != 1.0:
        output.mul_(routed_scaling_factor)

    return StandardCombineInput(hidden_states=output)
