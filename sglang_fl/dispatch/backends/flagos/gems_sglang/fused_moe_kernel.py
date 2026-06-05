"""Fused MoE v0 Triton-kernels implementation.

The public function mirrors SGLang's triton_kernel_fused_experts entrypoint so it
can replace the original path used by sglang-plugin-FL.
"""

from __future__ import annotations

from typing import Optional

import torch
from triton_kernels.matmul_ogs import matmul_ogs

try:
    from sglang.jit_kernel.activation import gelu_and_mul, silu_and_mul
except Exception:
    from sgl_kernel import gelu_and_mul, silu_and_mul

_UNSUPPORTED_FEATURE_MSG = "{name} is not supported"


def triton_kernel_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    routing_data,
    gather_indx,
    scatter_indx,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    assert use_fp8_w8a8 is False, _UNSUPPORTED_FEATURE_MSG.format(name="use_fp8_w8a8")
    assert per_channel_quant is False, _UNSUPPORTED_FEATURE_MSG.format(name="per_channel_quant")
    assert expert_map is None, _UNSUPPORTED_FEATURE_MSG.format(name="expert_map")
    assert w1_scale is None, _UNSUPPORTED_FEATURE_MSG.format(name="w1_scale")
    assert w2_scale is None, _UNSUPPORTED_FEATURE_MSG.format(name="w2_scale")
    assert a1_scale is None, _UNSUPPORTED_FEATURE_MSG.format(name="a1_scale")
    assert a2_scale is None, _UNSUPPORTED_FEATURE_MSG.format(name="a2_scale")
    assert block_shape is None, _UNSUPPORTED_FEATURE_MSG.format(name="block_shape")
    assert inplace is False, "Inplace is not supported"

    assert hidden_states.ndim == 2, "hidden_states must be 2D"
    assert hidden_states.dtype == torch.bfloat16, "hidden_states must be bfloat16"
    assert w1.dtype == torch.bfloat16, "w1 must be bfloat16"
    assert w2.dtype == torch.bfloat16, "w2 must be bfloat16"
    assert hidden_states.shape[-1] == w1.shape[-2], "hidden size mismatch"
    assert w2.shape[-1] == w1.shape[1], "intermediate size mismatch"

    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    inter_size_twice = w1.shape[2]
    top_k = routing_data.n_expts_act

    intermediate = matmul_ogs(
        hidden_states,
        w1,
        None,
        routing_data,
        gather_indx=gather_indx,
        gammas=routing_data.gate_scal if apply_router_weight_on_input else None,
    )

    activated = torch.empty(
        (num_tokens * top_k, inter_size_twice // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    if activation == "silu":
        silu_and_mul(intermediate.view(-1, inter_size_twice), activated)
    elif activation == "gelu":
        gelu_and_mul(intermediate.view(-1, inter_size_twice), activated)
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    output = matmul_ogs(
        activated,
        w2,
        None,
        routing_data,
        scatter_indx=scatter_indx,
        gammas=None if apply_router_weight_on_input else routing_data.gate_scal,
    )

    if scatter_indx is None:
        return output.view(num_tokens, top_k, hidden_size)
    return output


def triton_kernel_fused_experts_with_bias(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_pcg,
    b1: torch.Tensor,
    w2: torch.Tensor,
    w2_pcg,
    b2: torch.Tensor,
    routing_data,
    gather_indx,
    scatter_indx,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_clamp_limit: Optional[float] = None,
) -> torch.Tensor:
    from triton_kernels.matmul_ogs import FlexCtx, FnSpecs, FusedActivation, PrecisionConfig
    from triton_kernels.numerics import InFlexData
    from triton_kernels.swiglu import swiglu_fn

    assert use_fp8_w8a8 is False, _UNSUPPORTED_FEATURE_MSG.format(name="use_fp8_w8a8")
    assert per_channel_quant is False, _UNSUPPORTED_FEATURE_MSG.format(name="per_channel_quant")
    assert expert_map is None, _UNSUPPORTED_FEATURE_MSG.format(name="expert_map")
    assert w1_scale is None, _UNSUPPORTED_FEATURE_MSG.format(name="w1_scale")
    assert w2_scale is None, _UNSUPPORTED_FEATURE_MSG.format(name="w2_scale")
    assert a1_scale is None, _UNSUPPORTED_FEATURE_MSG.format(name="a1_scale")
    assert a2_scale is None, _UNSUPPORTED_FEATURE_MSG.format(name="a2_scale")
    assert block_shape is None, _UNSUPPORTED_FEATURE_MSG.format(name="block_shape")
    assert inplace is False, "Inplace is not supported"

    if w1_pcg is None:
        w1_pcg = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=InFlexData()))
    if w2_pcg is None:
        w2_pcg = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=InFlexData()))

    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    inter_size_twice = w1.shape[2]
    top_k = routing_data.n_expts_act
    act = FusedActivation(
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
        (gemm1_alpha, gemm1_clamp_limit),
    )

    activated = torch.empty(
        (1, num_tokens * top_k, inter_size_twice // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    output = torch.empty(
        (1, num_tokens, hidden_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    matmul_ogs(
        hidden_states,
        w1,
        b1,
        routing_data,
        gather_indx=gather_indx,
        precision_config=w1_pcg,
        gammas=routing_data.gate_scal if apply_router_weight_on_input else None,
        fused_activation=act,
        y=activated,
    )
    matmul_ogs(
        activated.view(num_tokens * top_k, inter_size_twice // 2),
        w2,
        b2,
        routing_data,
        scatter_indx=scatter_indx,
        precision_config=w2_pcg,
        gammas=None if apply_router_weight_on_input else routing_data.gate_scal,
        y=output,
    )
    return output.view(num_tokens, hidden_size)
