# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in Qwen3.6 TP2 adapter at the existing vendor fused_moe dispatch seam.

No core dataclass, post-load hook, parameter or runner registry is modified.
Check the loaded canonical weights at the call site instead of storing a
capability bit in SGLang's TritonMoeQuantInfo.
"""

import logging
import os
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)
_SELECTED_SHAPES = set()


def _enabled(name):
    return os.getenv(name, "0").strip().lower() in ("1", "true", "yes", "on")


@lru_cache(maxsize=8)
def _is_s5000(device):
    return "S5000" in str(torch.musa.get_device_name(device)).upper()


def _matches_layer(method, layer, server_args, a2a_backend):
    """The campaign's post-load eligibility check, without changing weights."""
    config = layer.moe_runner_config
    return (
        method._aiter_runner is None
        and method.runner.runner_backend.is_triton()
        and a2a_backend.is_none()
        and not method.with_bias
        and tuple(layer.w13_weight.shape) == (256, 512, 2048)
        and tuple(layer.w2_weight.shape) == (256, 2048, 256)
        and layer.w13_weight.dtype == torch.bfloat16
        and layer.w2_weight.dtype == torch.bfloat16
        and layer.w13_weight.is_contiguous()
        and layer.w2_weight.is_contiguous()
        and getattr(layer, "w13_weight_bias", None) is None
        and getattr(layer, "w2_weight_bias", None) is None
        and config.num_experts == 256
        and config.num_local_experts == 256
        and not config.num_fused_shared_experts
        and config.hidden_size == 2048
        and config.intermediate_size_per_partition == 256
        and config.top_k == 8
        and config.activation == "silu"
        and config.is_gated
        and not config.apply_router_weight_on_input
        and config.gemm1_alpha is None
        and config.gemm1_clamp_limit is None
        and not config.no_combine
        and getattr(layer, "moe_tp_size", None) == 2
        and not bool(getattr(server_args, "enable_lora", False))
        and not bool(getattr(server_args, "lora_paths", None))
        and not bool(getattr(server_args, "enable_eplb", False))
        and not bool(getattr(server_args, "enable_fused_moe_sum_all_reduce", False))
        and not bool(getattr(method.runner, "lora_enabled", False))
        and getattr(method.runner, "down_gemm_overlap_args", None) is None
        and getattr(method.runner, "meta_overlap_args", None) is None
        # Only the campaign's full-graph/eager path is validated here.
        and bool(getattr(server_args, "disable_piecewise_cuda_graph", False))
        and method.moe_runner_config is config
    )


def _matches_inputs(layer, hidden, topk_weights, topk_ids):
    return (
        hidden.device.type == "musa"
        and hidden.dtype == torch.bfloat16
        and hidden.ndim == 2
        and hidden.shape[0] > 0
        and hidden.shape[1] == 2048
        and hidden.is_contiguous()
        and tuple(topk_ids.shape) == (hidden.shape[0], 8)
        and topk_weights.shape == topk_ids.shape
        and topk_ids.dtype == torch.int32
        and topk_weights.dtype == torch.float32
        and topk_ids.is_contiguous()
        and topk_weights.is_contiguous()
        and all(
            tensor.device == hidden.device
            for tensor in (layer.w13_weight, layer.w2_weight, topk_weights, topk_ids)
        )
        and not torch.is_grad_enabled()
    )


def maybe_forward(method, layer, dispatch_output):
    """Return a StandardCombineInput on a hit; None leaves forward_musa intact."""
    fused = _enabled("SGLANG_MUSA_FUSE_MOE_SWIGLU_EPILOGUE")
    m4 = _enabled("SGLANG_MUSA_M4_W13_BN64")
    if not (fused or m4):
        return None
    hidden = getattr(dispatch_output, "hidden_states", None)
    if not isinstance(hidden, torch.Tensor) or hidden.device.type != "musa":
        return None
    if not fused and (hidden.ndim != 2 or hidden.shape[0] != 4):
        return None

    # Import only after the cheap device/feature guards. Unsupported versions
    # must retain their native vendor path, not partially initialize kernels.
    try:
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        from sglang.srt.layers.moe.topk import TopKOutputChecker
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
        from sglang.srt.server_args import get_global_server_args

        if type(method) is not UnquantizedFusedMoEMethod:
            return None
        if not TopKOutputChecker.format_is_standard(dispatch_output.topk_output):
            return None
        weights, ids, _ = dispatch_output.topk_output
        if not (
            _matches_layer(
                method, layer, get_global_server_args(), get_moe_a2a_backend()
            )
            and _matches_inputs(layer, hidden, weights, ids)
            and _is_s5000(hidden.device)
        ):
            return None
    except (ImportError, AttributeError, TypeError, ValueError):
        return None

    from .fused_moe import fused_experts

    # Do not catch a launch or allocation error here. A failed in-place kernel
    # cannot safely be retried against potentially modified hidden states.
    output = fused_experts(
        hidden_states=hidden,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_output=dispatch_output.topk_output,
        moe_runner_config=layer.moe_runner_config,
        fuse_swiglu_epilogue=fused,
    )
    selection = (hidden.shape[0], fused and hidden.shape[0] >= 2048)
    if selection not in _SELECTED_SHAPES:
        logger.info(
            "MUSA Qwen36 plugin-owned routed MoE selected: tokens=%s fused_swiglu=%s",
            *selection,
        )
        _SELECTED_SHAPES.add(selection)
    return StandardCombineInput(hidden_states=output)
