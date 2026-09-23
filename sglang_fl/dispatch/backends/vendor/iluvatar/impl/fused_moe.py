from __future__ import annotations

import torch


def fused_moe_iluvatar(
    obj,
    layer: torch.nn.Module,
    dispatch_output,
):
    cfg = getattr(layer, "moe_runner_config", None) or getattr(
        obj, "moe_runner_config", None
    )
    if (
        cfg is None
        or cfg.num_experts is None
        or cfg.num_local_experts is None
        or cfg.num_experts != cfg.num_local_experts
        or cfg.activation != "silu"
        or getattr(cfg, "apply_router_weight_on_input", False)
        or getattr(cfg, "gemm1_alpha", None) is not None
        or getattr(layer, "w13_weight_bias", None) is not None
        or getattr(layer, "w2_weight_bias", None) is not None
    ):
        raise NotImplementedError("unsupported Iluvatar fused MoE configuration")

    hidden = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    if (
        not hidden.is_cuda
        or hidden.dtype not in (torch.float16, torch.bfloat16)
        or hidden.dim() != 2
    ):
        raise NotImplementedError("Iluvatar fused MoE requires CUDA fp16/bf16 tensors")

    output = None
    if hidden.shape[0] >= 8 and not torch.cuda.is_current_stream_capturing():
        from .moe_ixformer import fused_experts, is_available

        if is_available():
            try:
                output = fused_experts(
                    hidden,
                    layer.w13_weight,
                    layer.w2_weight,
                    topk_weights,
                    topk_ids,
                )
            except (RuntimeError, TypeError, ValueError):
                output = None
    if output is None:
        from .moe_triton import fused_experts

        output = fused_experts(
            hidden.contiguous(),
            layer.w13_weight.contiguous(),
            layer.w2_weight.contiguous(),
            topk_weights,
            topk_ids,
        )

    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    return StandardCombineInput(hidden_states=output)
