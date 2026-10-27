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

"""FlagGems expert computation for the unquantized SGLang MoE bridge."""

from __future__ import annotations

import inspect

import torch


def is_fused_moe_available() -> bool:
    try:
        from flag_gems.fused import fused_experts_impl

        parameters = inspect.signature(fused_experts_impl).parameters
        return callable(fused_experts_impl) and all(
            name in parameters
            for name in (
                "apply_router_weight_on_input",
                "global_num_experts",
                "w1_bias",
                "w2_bias",
            )
        )
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def _validate_supported_config(obj, layer, dispatch_output) -> None:
    """Reject unsupported semantics before executing a kernel or mutating data."""
    config = obj.moe_runner_config
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if (
        layer.w13_weight.dtype not in supported_dtypes
        or layer.w2_weight.dtype not in supported_dtypes
        or dispatch_output.hidden_states.dtype not in supported_dtypes
        or getattr(obj, "quant_config", None) is not None
        or any(
            getattr(obj, name, False)
            for name in (
                "use_fp8_w8a8",
                "use_int8_w8a8",
                "use_int8_w8a16",
                "use_int4_w4a16",
                "use_mxfp8",
                "is_fp4_expert",
            )
        )
    ):
        raise NotImplementedError(
            "FlagOS fused_moe supports unquantized floating-point weights only"
        )
    if (
        layer.w13_weight.dtype != dispatch_output.hidden_states.dtype
        or layer.w2_weight.dtype != dispatch_output.hidden_states.dtype
    ):
        raise NotImplementedError(
            "FlagOS fused_moe requires matching activation and weight dtypes"
        )
    if layer.w13_weight.ndim != 3 or layer.w2_weight.ndim != 3:
        raise NotImplementedError("FlagOS fused_moe requires standard [E,N,K] weights")
    if (
        getattr(obj, "use_triton_kernels", False)
        or getattr(obj, "use_flashinfer_cutlass", False)
        or getattr(obj, "use_flashinfer_trtllm_moe", False)
        or getattr(obj, "use_deep_gemm", False)
        or getattr(obj, "w13_swiglu_interleaved", False)
    ):
        raise NotImplementedError(
            "FlagOS fused_moe requires standard split-gate [E,N,K] weights"
        )
    if (
        getattr(dispatch_output, "hidden_states_scale", None) is not None
        or getattr(dispatch_output, "hidden_states_pre_quant", None) is not None
    ):
        raise NotImplementedError(
            "FlagOS fused_moe does not support pre-quantized dispatch input"
        )
    if config.activation != "silu" or not config.is_gated:
        raise NotImplementedError("FlagOS fused_moe supports gated SiLU only")
    if config.no_combine:
        raise NotImplementedError("FlagOS fused_moe does not support no_combine")
    if getattr(config, "use_tp_all_gather_activation", False):
        raise NotImplementedError(
            "FlagOS fused_moe does not support TP activation all-gather"
        )
    if any(
        getattr(config, name, None) is not None
        for name in ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit", "swiglu_limit")
    ):
        raise NotImplementedError(
            "FlagOS fused_moe does not support custom SwiGLU alpha/beta/clamp"
        )
    if (
        config.num_experts is None
        or config.num_local_experts != config.num_experts
        or layer.w13_weight.shape[0] != config.num_experts
    ):
        raise NotImplementedError(
            "FlagOS fused_moe does not support expert-parallel mappings"
        )
    topk = dispatch_output.topk_output
    if not hasattr(topk, "topk_weights") or not hasattr(topk, "topk_ids"):
        raise NotImplementedError(
            "FlagOS fused_moe requires materialized standard top-k output"
        )
    hidden_size = dispatch_output.hidden_states.shape[-1]
    if (
        layer.w13_weight.shape[2] != hidden_size
        or layer.w13_weight.shape[1] != 2 * layer.w2_weight.shape[2]
        or layer.w2_weight.shape[1] != hidden_size
        or layer.w2_weight.shape[0] != config.num_experts
    ):
        raise NotImplementedError(
            "FlagOS fused_moe received an unsupported expert weight layout"
        )


def fused_moe_flagos(obj, layer: torch.nn.Module, dispatch_output):
    """Consume SGLang routing and return the standard combine representation."""
    _validate_supported_config(obj, layer, dispatch_output)
    from flag_gems.fused import fused_experts_impl
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    config = obj.moe_runner_config
    topk = dispatch_output.topk_output
    output = fused_experts_impl(
        hidden_states=dispatch_output.hidden_states.contiguous(),
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_weights=topk.topk_weights,
        topk_ids=topk.topk_ids.to(dtype=torch.int32),
        inplace=False,
        activation=config.activation,
        apply_router_weight_on_input=config.apply_router_weight_on_input,
        global_num_experts=config.num_experts,
        w1_bias=getattr(layer, "w13_weight_bias", None),
        w2_bias=getattr(layer, "w2_weight_bias", None),
    )
    factor = config.routed_scaling_factor
    if factor is not None and factor != 1.0:
        output.mul_(factor)
    return StandardCombineInput(hidden_states=output)
