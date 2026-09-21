# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Adapted from the pinned SGLang/MUSA Qwen3.6 campaign implementation.
# Upstream ancestry: vllm a6221a144af772fd1a68fe7e627935dc53e81738.
# See README.md for source hashes, boundaries and validation status.
"""Plugin-owned routed-MoE sequence; entered only by the exact MUSA adapter."""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import torch
import torch.nn.functional as F
import triton.language as tl
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe as _core
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.custom_op import register_custom_op
from .kernels import invoke_fused_moe_kernel

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import StandardTopKOutput

logger = logging.getLogger(__name__)
_is_musa = _core._is_musa
_is_cuda = _is_hip = _is_xpu = _use_aiter = _has_vllm_ops = False
padding_size = _core.padding_size
_silu_and_mul_musa = getattr(_core, "_silu_and_mul_musa", None)
# Non-MUSA branches remain in the source-fidelity copy but cannot be selected
# by dispatch.py. Keep their optional symbols explicit; do not import another
# vendor implementation merely to initialize this MUSA module.
silu_and_mul = getattr(_core, "silu_and_mul", None)
gelu_and_mul = getattr(_core, "gelu_and_mul", None)
moe_sum = getattr(_core, "moe_sum", None)
moe_sum_reduce_triton = getattr(_core, "moe_sum_reduce_triton", None)
vllm_ops = getattr(_core, "vllm_ops", None)


# Resolve through the original module at call time. In particular, never cache
# moe_sum_reduce: the deterministic combine patch and its ContextVar own it.
def try_get_optimal_moe_config(*args, **kwargs):
    return _core.try_get_optimal_moe_config(*args, **kwargs)


def get_config_dtype_str(*args, **kwargs):
    return _core.get_config_dtype_str(*args, **kwargs)


def moe_align_block_size(*args, **kwargs):
    return _core.moe_align_block_size(*args, **kwargs)


def _down_moe_use_tma():
    return _core._down_moe_use_tma()


def moe_sum_reduce(*args, **kwargs):
    return _core.moe_sum_reduce(*args, **kwargs)


def moe_sum_reduce_torch_compile(*args, **kwargs):
    return _core.moe_sum_reduce_torch_compile(*args, **kwargs)


def _swiglu_silu_clamp_mul(*args, **kwargs):
    return _core._swiglu_silu_clamp_mul(*args, **kwargs)


def swiglu_gpt_oss_sigmoid_alpha(*args, **kwargs):
    return _core.swiglu_gpt_oss_sigmoid_alpha(*args, **kwargs)


# Product marker for the exact Qwen3.6 TP2 prefill schedule.  The resolver's
# dictionary is copied at launch time; the original object remains the source
# of route alignment and the down-GEMM schedule.
_MUSA_FUSED_SWIGLU_UP_BLOCK_SIZE_K = 32
_MUSA_FUSED_SWIGLU_DOWN_NUM_WARPS = 16
_MUSA_FUSED_SWIGLU_DOWN_PREFILL_TOKENS = frozenset((8192, 16384))
_MUSA_M16K_MOE_DOWN_WORKSPACE_ENV = "SGLANG_MUSA_M16K_MOE_PREALLOCATE_DOWN_WORKSPACE"
_MUSA_M4_W13_BN64_ENV = "SGLANG_MUSA_M4_W13_BN64"
_MUSA_M16K_MOE_DOWN_WORKSPACE: Optional[torch.Tensor] = None
_MUSA_M16K_MOE_DOWN_WORKSPACE_USED = False


def _maybe_select_musa_m4_w13_bn64_config(
    config: Dict[str, Any],
    *,
    enabled: bool,
    is_musa: bool,
    num_tokens: int,
    hidden_shape: tuple[int, ...],
    w1_shape: tuple[int, ...],
    w2_shape: tuple[int, ...],
    topk_shape: tuple[int, ...],
    hidden_bf16: bool,
    weights_bf16: bool,
    topk_int32: bool,
    same_device: bool,
    tensors_contiguous: bool,
    b1_absent: bool,
    b2_absent: bool,
    use_quantization: bool,
    per_channel_quant: bool,
    block_shape: Optional[List[int]],
    activation: str,
    is_gated: bool,
    no_combine: bool,
    inplace: bool,
    apply_router_weight_on_input: bool,
    routed_scaling_factor: Optional[float],
    gemm1_alpha: Optional[float],
    gemm1_limit: Optional[float],
    filter_expert: bool,
    hooks_absent: bool,
    down_moe_use_tma: bool,
    effective_fused_swiglu: bool,
    fused_moe_sum_all_reduce: bool,
    grad_disabled: bool,
) -> Dict[str, Any]:
    """Return an opt-in M4 W13 tile copy without changing the down config.

    The selector is deliberately stricter than the kernel's general dtype and
    routing contract.  It is a source-only diagnostic for the exact BF16
    Qwen M4 shape; all unsupported features retain the resolver's object.
    Only ``BLOCK_SIZE_N`` is changed in the returned copy, so explicit or
    implicit warp/stage values and any future config keys survive unchanged.
    """
    exact_shape = (
        is_musa
        and num_tokens == 4
        and hidden_shape == (4, 2048)
        and w1_shape == (256, 512, 2048)
        and w2_shape == (256, 2048, 256)
        and topk_shape == (4, 8)
        and hidden_bf16
        and weights_bf16
        and topk_int32
        and same_device
        and tensors_contiguous
    )
    exact_features = (
        enabled
        and b1_absent
        and b2_absent
        and not use_quantization
        and not per_channel_quant
        and block_shape is None
        and activation == "silu"
        and is_gated
        and not no_combine
        and inplace
        and not apply_router_weight_on_input
        and routed_scaling_factor in (None, 1.0)
        and gemm1_alpha is None
        and gemm1_limit is None
        and not filter_expert
        and hooks_absent
        and not down_moe_use_tma
        and not effective_fused_swiglu
        and not fused_moe_sum_all_reduce
        and grad_disabled
    )
    baseline_tile = (
        config.get("BLOCK_SIZE_M") == 16
        and config.get("BLOCK_SIZE_N") == 32
        and config.get("BLOCK_SIZE_K") == 64
        and config.get("GROUP_SIZE_M") == 1
        and not config.get("USE_TMA", False)
    )
    if not (exact_shape and exact_features and baseline_tile):
        return config

    selected = dict(config)
    selected["BLOCK_SIZE_N"] = 64
    return selected


def maybe_preallocate_musa_m16k_moe_down_workspace(
    model_config: Any,
    device: Any,
    *,
    tp_size: int,
    moe_ep_size: int,
    is_draft_worker: bool,
) -> bool:
    """Reserve the exact M16K routed-down output before sizing the KV pool.

    The persistent allocation follows the same lifetime ordering used by
    upstream workspace managers: account for process-lifetime scratch before
    the large KV backing allocation, then reuse a fixed-address tensor across
    sequential MoE layers and graph replays.  Keep the experiment opt-in and
    exact-model gated while its capacity and graph contracts are validated.
    """
    global _MUSA_M16K_MOE_DOWN_WORKSPACE

    if not get_bool_env_var(_MUSA_M16K_MOE_DOWN_WORKSPACE_ENV):
        return False

    text_config = getattr(model_config, "hf_text_config", None)
    rope_config = getattr(text_config, "rope_parameters", None) or getattr(
        text_config, "rope_scaling", None
    )
    mrope_section = (
        rope_config.get("mrope_section") if isinstance(rope_config, dict) else None
    )
    matches = (
        _is_musa
        and not is_draft_worker
        and tp_size == 2
        and moe_ep_size == 1
        and getattr(model_config, "dtype", None) == torch.bfloat16
        and getattr(model_config, "vocab_size", None) == 248320
        and getattr(text_config, "model_type", None) == "qwen3_5_moe_text"
        and getattr(text_config, "hidden_size", None) == 2048
        and getattr(text_config, "num_hidden_layers", None) == 40
        and getattr(text_config, "num_experts", None) == 256
        and getattr(text_config, "num_experts_per_tok", None) == 8
        and getattr(text_config, "moe_intermediate_size", None) == 512
        and getattr(text_config, "shared_expert_intermediate_size", None) == 512
        and tuple(mrope_section or ()) == (11, 11, 10)
    )
    if not matches:
        logger.warning(
            "%s ignored because the loaded model/runtime is outside the exact "
            "Qwen3.6 V248320 MRoPE TP2 BF16 contract",
            _MUSA_M16K_MOE_DOWN_WORKSPACE_ENV,
        )
        return False

    if _MUSA_M16K_MOE_DOWN_WORKSPACE is None:
        _MUSA_M16K_MOE_DOWN_WORKSPACE = torch.empty(
            (16384, 8, 2048),
            dtype=torch.bfloat16,
            device=device,
        )
        logger.info(
            "Pre-allocated exact M16K routed-MoE down workspace before KV pool: "
            "bytes=%s",
            _MUSA_M16K_MOE_DOWN_WORKSPACE.numel()
            * _MUSA_M16K_MOE_DOWN_WORKSPACE.element_size(),
        )
    return True


def _get_musa_m16k_moe_down_workspace(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    b1: Optional[torch.Tensor],
    b2: Optional[torch.Tensor],
    use_fused_swiglu_epilogue: bool,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    w1_zp: Optional[torch.Tensor],
    w2_zp: Optional[torch.Tensor],
    a1_scale: Optional[torch.Tensor],
    a2_scale: Optional[torch.Tensor],
    block_shape: Optional[List[int]],
    activation: str,
    is_gated: bool,
    no_combine: bool,
    inplace: bool,
    apply_router_weight_on_input: bool,
    routed_scaling_factor: Optional[float],
    gemm1_alpha: Optional[float],
    gemm1_limit: Optional[float],
    filter_expert: bool,
    hooks: Optional[Any],
) -> Optional[torch.Tensor]:
    """Return the pre-reserved tensor only for the proven allocation shape."""
    workspace = _MUSA_M16K_MOE_DOWN_WORKSPACE
    if (
        workspace is None
        or hidden_states.ndim != 2
        or hidden_states.shape[0] != 16384
        or hidden_states.shape[1] != 2048
    ):
        return None

    tensors = (hidden_states, w1, w2, topk_weights, topk_ids, workspace)
    if not (
        _is_musa
        and tuple(w1.shape) == (256, 512, 2048)
        and tuple(w2.shape) == (256, 2048, 256)
        and tuple(topk_weights.shape) == (16384, 8)
        and tuple(topk_ids.shape) == (16384, 8)
        and tuple(workspace.shape) == (16384, 8, 2048)
        and hidden_states.dtype == torch.bfloat16
        and w1.dtype == torch.bfloat16
        and w2.dtype == torch.bfloat16
        and topk_weights.dtype == torch.float32
        and topk_ids.dtype == torch.int32
        and workspace.dtype == torch.bfloat16
        and all(tensor.device == hidden_states.device for tensor in tensors)
        and all(tensor.is_contiguous() for tensor in tensors)
        and b1 is None
        and b2 is None
        and use_fused_swiglu_epilogue
        and not (use_fp8_w8a8 or use_int8_w8a8 or use_int8_w8a16 or use_int4_w4a16)
        and not per_channel_quant
        and all(
            value is None
            for value in (
                w1_scale,
                w2_scale,
                w1_zp,
                w2_zp,
                a1_scale,
                a2_scale,
                block_shape,
            )
        )
        and activation == "silu"
        and is_gated
        and not no_combine
        and inplace
        and not apply_router_weight_on_input
        and routed_scaling_factor in (None, 1.0)
        and gemm1_alpha is None
        and gemm1_limit is None
        and not filter_expert
        and hooks is None
    ):
        return None
    return workspace


def _maybe_select_musa_fused_swiglu_down_config(
    down_config: Optional[Dict[str, Any]],
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    fuse_swiglu_epilogue: bool,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: Optional[List[int]],
) -> Optional[Dict[str, Any]]:
    """Select the measured W16 down schedule for the exact fused prefill.

    This is intentionally a conservative selector seam.  It only copies the
    resolver's down dictionary when the canonical Qwen3.6 TP2 BF16 shape and
    the already-supported fused path are present, the token count is one of
    the measured M8192/M16384 values, and the resolver supplied the expected
    W8 baseline.  Any unsupported feature, different resolver, or adjacent
    shape falls through to the resolver's original dictionary unchanged.
    """
    if down_config is None:
        return down_config
    if not (
        fuse_swiglu_epilogue
        and _is_musa
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.ndim == 2
        and hidden_states.shape[1] == 2048
        and hidden_states.shape[0] in _MUSA_FUSED_SWIGLU_DOWN_PREFILL_TOKENS
        and tuple(w1.shape) == (256, 512, 2048)
        and tuple(w2.shape) == (256, 2048, 256)
        and w1.dtype == torch.bfloat16
        and w2.dtype == torch.bfloat16
        and w1.device == hidden_states.device == w2.device
        and hidden_states.is_contiguous()
        and w1.is_contiguous()
        and w2.is_contiguous()
        and topk_ids.ndim == 2
        and topk_ids.shape[0] == hidden_states.shape[0]
        and topk_ids.shape[1] == 8
        and topk_ids.dtype == torch.int32
        and topk_ids.device == hidden_states.device
        and topk_ids.is_contiguous()
        and not (use_fp8_w8a8 or use_int8_w8a8 or use_int8_w8a16 or use_int4_w4a16)
        and not per_channel_quant
        and block_shape is None
        and not down_config.get("USE_TMA", False)
    ):
        return down_config

    expected_block_m = 32 if hidden_states.shape[0] == 8192 else 64
    expected = {
        "BLOCK_SIZE_M": expected_block_m,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 4,
        "num_warps": 8,
        "num_stages": 1,
    }
    if any(down_config.get(key) != value for key, value in expected.items()):
        return down_config

    selected = dict(down_config)
    selected["num_warps"] = _MUSA_FUSED_SWIGLU_DOWN_NUM_WARPS
    return selected


@register_custom_op(
    op_name="musa_qwen36_inplace_fused_experts", mutates_args=["hidden_states"]
)
def inplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    activation: str = "silu",
    is_gated: bool = True,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    filter_expert: bool = True,
    fuse_swiglu_epilogue: bool = False,
) -> None:
    fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        b1,
        b2,
        True,
        activation,
        is_gated,
        apply_router_weight_on_input,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        False,
        routed_scaling_factor,
        gemm1_alpha,
        gemm1_limit,
        filter_expert,
        fuse_swiglu_epilogue=fuse_swiglu_epilogue,
    )


@register_custom_op(
    op_name="musa_qwen36_outplace_fused_experts", out_shape="hidden_states"
)
def outplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    activation: str = "silu",
    is_gated: bool = True,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    filter_expert: bool = True,
    fuse_swiglu_epilogue: bool = False,
) -> torch.Tensor:
    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        b1,
        b2,
        False,
        activation,
        is_gated,
        apply_router_weight_on_input,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        no_combine=no_combine,
        routed_scaling_factor=routed_scaling_factor,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
        filter_expert=filter_expert,
        fuse_swiglu_epilogue=fuse_swiglu_epilogue,
    )


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_output: StandardTopKOutput,
    moe_runner_config: MoeRunnerConfig,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    fuse_swiglu_epilogue: bool = False,
):
    topk_weights, topk_ids, _ = topk_output
    filter_expert = (
        moe_runner_config.num_experts is None
        or moe_runner_config.num_experts != moe_runner_config.num_local_experts
    )
    if moe_runner_config.inplace:
        assert not moe_runner_config.no_combine, "no combine + inplace makes no sense"
        inplace_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            b1,
            b2,
            moe_runner_config.activation,
            moe_runner_config.is_gated,
            moe_runner_config.apply_router_weight_on_input,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            w1_scale,
            w2_scale,
            w1_zp,
            w2_zp,
            a1_scale,
            a2_scale,
            block_shape,
            moe_runner_config.routed_scaling_factor,
            moe_runner_config.gemm1_alpha,
            moe_runner_config.gemm1_clamp_limit,
            filter_expert,
            fuse_swiglu_epilogue=fuse_swiglu_epilogue,
        )
        return hidden_states
    else:
        return outplace_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            b1,
            b2,
            moe_runner_config.activation,
            moe_runner_config.is_gated,
            moe_runner_config.apply_router_weight_on_input,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            w1_scale,
            w2_scale,
            w1_zp,
            w2_zp,
            a1_scale,
            a2_scale,
            block_shape,
            no_combine=moe_runner_config.no_combine,
            routed_scaling_factor=moe_runner_config.routed_scaling_factor,
            gemm1_alpha=moe_runner_config.gemm1_alpha,
            gemm1_limit=moe_runner_config.gemm1_clamp_limit,
            filter_expert=filter_expert,
            fuse_swiglu_epilogue=fuse_swiglu_epilogue,
        )


def _prepare_fused_moe_run(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: Optional[List[int]],
    fuse_swiglu_epilogue: bool = False,
):
    """Resolve config, down_config, TMA flag, and aligned expert routing ids.

    Shared by ``fused_experts_impl`` and ``pre_permute_standard_to_triton`` so
    both paths compute alignment from the same source.
    """
    padded_size = padding_size
    if not (use_fp8_w8a8 or use_int8_w8a8) or block_shape is not None or _use_aiter:
        padded_size = 0

    num_tokens = hidden_states.shape[0]
    E = w1.shape[0]
    config_dtype = get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        dtype=hidden_states.dtype,
    )

    config, (down_config, _) = try_get_optimal_moe_config(
        w1.shape,
        (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),
        topk_ids.shape[1],
        config_dtype,
        num_tokens,
        block_shape=block_shape,
        per_channel_quant=per_channel_quant,
        return_down_config=True,
    )
    down_config = _maybe_select_musa_fused_swiglu_down_config(
        down_config,
        hidden_states,
        w1,
        w2,
        topk_ids,
        fuse_swiglu_epilogue=fuse_swiglu_epilogue,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
    )
    down_moe_use_tma = (
        _down_moe_use_tma()
        and down_config is not None
        and down_config.pop("USE_TMA", False)
    )

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E
    )

    return (
        config,
        down_config,
        down_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    )


def _fused_moe_kernel_sequence(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: Dict[str, Any],
    down_config: Optional[Dict[str, Any]],
    down_moe_use_tma: bool,
    *,
    b1: Optional[torch.Tensor],
    b2: Optional[torch.Tensor],
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    w1_zp: Optional[torch.Tensor],
    w2_zp: Optional[torch.Tensor],
    a1_scale: Optional[torch.Tensor],
    a2_scale: Optional[torch.Tensor],
    block_shape: Optional[List[int]],
    activation: str,
    is_gated: bool,
    no_combine: bool,
    inplace: bool,
    apply_router_weight_on_input: bool,
    routed_scaling_factor: Optional[float],
    gemm1_alpha: Optional[float],
    gemm1_limit: Optional[float],
    filter_expert: bool,
    hooks: Optional[Any] = None,
    fuse_swiglu_epilogue: bool = False,
) -> torch.Tensor:
    """Run the MoE kernel/activation/kernel/combine sequence in a single shot.

    Inputs are already aligned and the block-size config is already resolved.
    Supports optional LoRA hooks that fire between the two kernels and before
    combine. Returns ``out_hidden_states``.
    """
    global _MUSA_M16K_MOE_DOWN_WORKSPACE_USED

    num_tokens = hidden_states.shape[0]
    # The fused path is intentionally a prefill-only optimization.  Keeping
    # the capability bit in quant_info but applying the gate here means that
    # decode (and any small batch) continues to consume canonical W13 through
    # the ordinary full-width path.
    use_fused_swiglu_epilogue = fuse_swiglu_epilogue and num_tokens >= 2048
    E, N, _ = w1.shape
    topk = topk_ids.shape[1]
    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

    padded_tokens = (
        min(num_tokens * topk, E + 1) * (config["BLOCK_SIZE_M"] - 1)
        if down_moe_use_tma
        else 0
    )
    total_tokens = num_tokens * topk + padded_tokens

    if no_combine:
        assert not inplace
        out_hidden_states = torch.empty(
            (num_tokens, topk, w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    elif inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    use_fused_moe_sum_all_reduce = (
        get_global_server_args().enable_fused_moe_sum_all_reduce
        and (not no_combine)
        and (topk > 2)
        and (not use_int8_w8a16)
        and (not use_int4_w4a16)
    )

    if use_fused_swiglu_epilogue:
        assert (
            _is_musa
            and tuple(w1.shape[1:]) == (512, 2048)
            and tuple(w2.shape[1:]) == (2048, 256)
            and w1.shape[0] == 256
            and topk == 8
            and activation == "silu"
            and is_gated
            and gemm1_alpha is None
            and gemm1_limit is None
            and b1 is None
            and b2 is None
            and not (use_fp8_w8a8 or use_int8_w8a8 or use_int8_w8a16 or use_int4_w4a16)
            and not apply_router_weight_on_input
            and not no_combine
            and not filter_expert
            and not down_moe_use_tma
            and not use_fused_moe_sum_all_reduce
            and hooks is None
            and hidden_states.dtype == torch.bfloat16
        ), "fuse_swiglu_epilogue reached an incompatible MoE call"
        intermediate_cache1 = None
        gemm1_out = intermediate_cache2 = torch.empty(
            (total_tokens, N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    else:
        gemm1_out = intermediate_cache1 = torch.empty(
            (total_tokens, N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    # The tested fused arm needs BK32, while the recorded product resolver
    # still returns BK64 for this exact prefill contract.  Copy only the up
    # config after the exact feature assertions: alignment already used the
    # resolver's original BLOCK_SIZE_M, and the down path below deliberately
    # continues to use ``down_config or config``.
    up_config = config
    if use_fused_swiglu_epilogue:
        up_config = dict(config)
        up_config["BLOCK_SIZE_K"] = _MUSA_FUSED_SWIGLU_UP_BLOCK_SIZE_K
    elif _is_musa and num_tokens == 4 and get_bool_env_var(_MUSA_M4_W13_BN64_ENV):
        up_config = _maybe_select_musa_m4_w13_bn64_config(
            config,
            enabled=True,
            is_musa=_is_musa,
            num_tokens=num_tokens,
            hidden_shape=tuple(hidden_states.shape),
            w1_shape=tuple(w1.shape),
            w2_shape=tuple(w2.shape),
            topk_shape=tuple(topk_ids.shape),
            hidden_bf16=hidden_states.dtype == torch.bfloat16,
            weights_bf16=(w1.dtype == torch.bfloat16 and w2.dtype == torch.bfloat16),
            topk_int32=topk_ids.dtype == torch.int32,
            same_device=(
                hidden_states.device == w1.device
                and hidden_states.device == w2.device
                and hidden_states.device == topk_ids.device
            ),
            tensors_contiguous=(
                hidden_states.is_contiguous()
                and w1.is_contiguous()
                and w2.is_contiguous()
                and topk_ids.is_contiguous()
            ),
            b1_absent=b1 is None,
            b2_absent=b2 is None,
            use_quantization=(
                use_fp8_w8a8 or use_int8_w8a8 or use_int8_w8a16 or use_int4_w4a16
            ),
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            activation=activation,
            is_gated=is_gated,
            no_combine=no_combine,
            inplace=inplace,
            apply_router_weight_on_input=apply_router_weight_on_input,
            routed_scaling_factor=routed_scaling_factor,
            gemm1_alpha=gemm1_alpha,
            gemm1_limit=gemm1_limit,
            filter_expert=filter_expert,
            hooks_absent=hooks is None,
            down_moe_use_tma=down_moe_use_tma,
            effective_fused_swiglu=use_fused_swiglu_epilogue,
            fused_moe_sum_all_reduce=use_fused_moe_sum_all_reduce,
            grad_disabled=not torch.is_grad_enabled(),
        )

    invoke_fused_moe_kernel(
        hidden_states,
        w1,
        b1,
        gemm1_out,
        a1_scale,
        w1_scale,
        w1_zp,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        apply_router_weight_on_input,
        topk,
        up_config,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        c_sorted=down_moe_use_tma,
        filter_expert=filter_expert,
        fuse_swiglu=use_fused_swiglu_epilogue,
    )

    if hooks and hooks.after_gate_up:
        # Hooks expect intermediate_cache1 shaped (num_tokens, topk, N); the
        # underlying buffer is laid out as (total_tokens, N) where
        # total_tokens = num_tokens * topk (+ TMA padding). Slice off any
        # padding and reshape for the hook, which writes in-place on the view.
        hooks.after_gate_up(
            hidden_states,
            intermediate_cache1[: num_tokens * topk].view(num_tokens, topk, N),
            topk_weights,
            topk_ids,
        )

    if not use_fused_swiglu_epilogue:
        intermediate_cache2 = torch.empty(
            (total_tokens, N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    # Activation function with multiplication
    if use_fused_swiglu_epilogue:
        pass
    elif activation == "silu" and is_gated:
        # - gemm1_alpha != None: GPT-OSS-style swiglu(alpha, limit)
        # - gemm1_alpha == None and gemm1_limit != None: silu+clamp+mul(limit-only)
        if gemm1_alpha is not None:
            assert gemm1_limit is not None
            intermediate_cache2 = swiglu_gpt_oss_sigmoid_alpha(
                intermediate_cache1.view(-1, N), gemm1_alpha, gemm1_limit
            )
        elif gemm1_limit is not None:
            intermediate_cache2 = _swiglu_silu_clamp_mul(
                intermediate_cache1.view(-1, N), gemm1_limit
            )
        elif _is_cuda or _is_hip or _is_xpu:
            if filter_expert and _is_cuda:
                # HIP/XPU fall through to the unfiltered path: the down kernel
                # zeros filtered rows without reading their input.
                silu_and_mul(
                    intermediate_cache1.view(-1, N),
                    intermediate_cache2,
                    expert_ids=(expert_ids if down_moe_use_tma else topk_ids.view(-1)),
                    expert_step=(config["BLOCK_SIZE_M"] if down_moe_use_tma else 1),
                )
            else:
                silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
        elif _is_musa:
            intermediate_cache2 = _silu_and_mul_musa(intermediate_cache1.view(-1, N))
        else:
            if _has_vllm_ops:
                vllm_ops.silu_and_mul(
                    intermediate_cache2, intermediate_cache1.view(-1, N)
                )
            else:
                # Fallback: native PyTorch silu_and_mul
                x = intermediate_cache1.view(-1, N)
                d = x.shape[-1] // 2
                intermediate_cache2.copy_(F.silu(x[..., :d]) * x[..., d:])
    elif activation == "gelu" and is_gated:
        assert gemm1_alpha is None, "gemm1_alpha is not supported for gelu"
        assert gemm1_limit is None, "gemm1_limit is not supported for gelu"
        if _is_cuda or _is_hip:
            if filter_expert and _is_cuda:
                gelu_and_mul(
                    intermediate_cache1.view(-1, N),
                    intermediate_cache2,
                    expert_ids=(expert_ids if down_moe_use_tma else topk_ids.view(-1)),
                    expert_step=(config["BLOCK_SIZE_M"] if down_moe_use_tma else 1),
                )
            else:
                gelu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
        else:
            if _has_vllm_ops:
                vllm_ops.gelu_and_mul(
                    intermediate_cache2, intermediate_cache1.view(-1, N)
                )
            else:
                # Fallback: native PyTorch gelu_and_mul
                x = intermediate_cache1.view(-1, N)
                d = x.shape[-1] // 2
                intermediate_cache2.copy_(F.gelu(x[..., :d]) * x[..., d:])
    # Activation function without multiplication
    elif activation == "silu" and not is_gated:
        intermediate_cache2 = F.silu(intermediate_cache1.view(-1, N))
    elif activation == "gelu" and not is_gated:
        intermediate_cache2 = F.gelu(intermediate_cache1.view(-1, N))
    elif activation == "relu2" and not is_gated:
        intermediate_cache2 = torch.square(F.relu(intermediate_cache1.view(-1, N)))
    else:
        raise ValueError(f"Unsupported activation: {activation=}, with {is_gated=}")

    del intermediate_cache1

    intermediate_cache3 = _get_musa_m16k_moe_down_workspace(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        b1=b1,
        b2=b2,
        use_fused_swiglu_epilogue=use_fused_swiglu_epilogue,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        activation=activation,
        is_gated=is_gated,
        no_combine=no_combine,
        inplace=inplace,
        apply_router_weight_on_input=apply_router_weight_on_input,
        routed_scaling_factor=routed_scaling_factor,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
        filter_expert=filter_expert,
        hooks=hooks,
    )
    if intermediate_cache3 is None:
        intermediate_cache3 = torch.empty(
            (num_tokens, topk, w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    elif not _MUSA_M16K_MOE_DOWN_WORKSPACE_USED:
        logger.info("Exact M16K routed-MoE down workspace reuse reached")
        _MUSA_M16K_MOE_DOWN_WORKSPACE_USED = True

    # LoRA hooks force the second kernel to write to intermediate_cache3 so
    # hooks.after_down can inspect/modify it before reduction.
    _use_intermediate = not no_combine and (topk != 1 or hooks)

    out_slice = None
    if use_fused_moe_sum_all_reduce:
        out_slice = out_hidden_states
        out_slice.zero_()

    invoke_fused_moe_kernel(
        intermediate_cache2,
        w2,
        b2,
        (
            out_slice
            if use_fused_moe_sum_all_reduce
            else (
                intermediate_cache3
                if _use_intermediate
                else out_hidden_states.unsqueeze(0)
            )
        ),
        a2_scale,
        w2_scale,
        w2_zp,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not apply_router_weight_on_input and not no_combine,
        1,
        down_config or config,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        a_use_tma=down_moe_use_tma,
        b_use_tma=down_moe_use_tma,
        filter_expert=filter_expert,
        fuse_sum_all_reduce=use_fused_moe_sum_all_reduce,
        router_topk=topk,
    )

    if hooks and hooks.after_down:
        hooks.after_down(
            intermediate_cache2, intermediate_cache3, topk_weights, topk_ids
        )

    del intermediate_cache2

    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    if no_combine:
        pass
    elif _is_cuda or _is_musa:
        if use_fused_moe_sum_all_reduce:
            if routed_scaling_factor != 1.0:
                assert out_slice is not None
                out_slice.mul_(routed_scaling_factor)
        elif topk == 1 and routed_scaling_factor == 1.0 and not _use_intermediate:
            pass  # we wrote directly into out_hidden_states
        elif topk == 2 and routed_scaling_factor == 1.0:
            torch.add(
                intermediate_cache3[:, 0],
                intermediate_cache3[:, 1],
                out=out_hidden_states,
            ).squeeze(dim=1)
        else:
            # According to micro benchmark results, torch.compile can get better performance for small token.
            if num_tokens <= 32:
                moe_sum_reduce_torch_compile(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states,
                    routed_scaling_factor,
                )
            else:
                moe_sum_reduce(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states,
                    routed_scaling_factor,
                )
    elif _is_hip:
        if _use_aiter:
            moe_sum(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states,
            )
        else:
            # According to micro benchmark results, torch.compile can get better performance for small token.
            if num_tokens <= 32:
                moe_sum_reduce_torch_compile(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states,
                    routed_scaling_factor,
                )
            else:
                moe_sum_reduce_triton(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states,
                    routed_scaling_factor,
                )
    elif _is_xpu:
        moe_sum_reduce(
            intermediate_cache3.view(*intermediate_cache3.shape),
            out_hidden_states,
            routed_scaling_factor,
        )
    else:
        if _has_vllm_ops:
            vllm_ops.moe_sum(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states,
            )
        else:
            # Fallback: use triton moe_sum_reduce when vllm is not available
            moe_sum_reduce_triton(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states,
                routed_scaling_factor,
            )

    del intermediate_cache3

    return out_hidden_states


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    inplace: bool = False,
    activation: str = "silu",
    is_gated: bool = True,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    filter_expert: bool = True,
    fuse_swiglu_epilogue: bool = False,
):
    padded_size = padding_size
    if not (use_fp8_w8a8 or use_int8_w8a8) or block_shape is not None or _use_aiter:
        padded_size = 0

    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.shape[1] // 2 == w1.shape[2], "Hidden size mismatch"
    else:
        assert (
            hidden_states.shape[1] == w1.shape[2] - padded_size
        ), f"Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    (
        config,
        down_config,
        down_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    ) = _prepare_fused_moe_run(
        hidden_states,
        w1,
        w2,
        topk_ids,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        fuse_swiglu_epilogue=fuse_swiglu_epilogue,
    )

    return _fused_moe_kernel_sequence(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
        down_config,
        down_moe_use_tma,
        b1=b1,
        b2=b2,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        activation=activation,
        is_gated=is_gated,
        no_combine=no_combine,
        inplace=inplace,
        apply_router_weight_on_input=apply_router_weight_on_input,
        routed_scaling_factor=routed_scaling_factor,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
        filter_expert=filter_expert,
        hooks=None,
        fuse_swiglu_epilogue=fuse_swiglu_epilogue,
    )
