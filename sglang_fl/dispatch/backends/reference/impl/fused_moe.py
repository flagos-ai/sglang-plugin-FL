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

# Reference PyTorch fused MoE (unquantized), CUDA-graph friendly.
#
# Mirrors SGLang ``fused_moe_forward_native`` / kunlunxin ``klx_fused_experts``
# contract: routing is already done; we only run expert MLP + weighted combine.
#
# Weight layout (UnquantizedFusedMoEMethod):
#   w13_weight [E, 2*inter, hidden]  (gate | up when is_gated)
#   w2_weight  [E, hidden, inter]
#
# Memory: process one top-k slot and a fixed token chunk at a time so the
# gathered weight tensor stays small (needed for long multimodal prefill).
# Loop bounds come from tensor shapes (fixed during CUDA Graph capture).

from __future__ import annotations

import torch
from torch.nn import functional as F

# Tokens per gather. 64 * 1024 * 2048 * 2B ≈ 256 MiB for a full w13 chunk.
_TOKEN_CHUNK = 64


def _activate(x_gate: torch.Tensor, x_up: torch.Tensor | None, activation: str) -> torch.Tensor:
    if x_up is None:
        if activation == "silu":
            return F.silu(x_gate)
        if activation == "gelu":
            return F.gelu(x_gate)
        raise ValueError(f"Unsupported activation: {activation=}")
    if activation == "silu":
        return F.silu(x_gate) * x_up
    if activation == "gelu":
        return F.gelu(x_gate) * x_up
    raise ValueError(f"Unsupported gated activation: {activation=}")


def fused_moe_torch(
    obj,
    layer: torch.nn.Module,
    dispatch_output,
):
    """Unquantized fused MoE in pure PyTorch (chunked gather + einsum).

    Graph-safe: no ``.item()`` / CPU sync. Top-k and token-chunk loops are
    bounded by tensor shapes, which are static during CUDA Graph capture.

    Args:
        obj: ``UnquantizedFusedMoEMethod`` (holds ``moe_runner_config``).
        layer: MoE layer with ``w13_weight`` / ``w2_weight``.
        dispatch_output: ``StandardDispatchOutput``.

    Returns:
        ``StandardCombineInput(hidden_states=...)``.
    """
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    x = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    topk_weights, topk_ids, *_ = topk_output

    cfg = getattr(obj, "moe_runner_config", None) or getattr(
        layer, "moe_runner_config", None
    )
    if cfg is None:
        raise RuntimeError("fused_moe_torch: moe_runner_config not found on obj/layer")

    if getattr(cfg, "apply_router_weight_on_input", False):
        raise NotImplementedError(
            "fused_moe_torch: apply_router_weight_on_input is not supported"
        )

    activation = getattr(cfg, "activation", "silu")
    is_gated = getattr(cfg, "is_gated", True)

    num_tokens, hidden = x.shape
    num_topk = topk_ids.shape[1]
    out = torch.zeros(num_tokens, hidden, device=x.device, dtype=x.dtype)

    w13_weight = layer.w13_weight
    w2_weight = layer.w2_weight
    w13_bias = getattr(layer, "w13_weight_bias", None)
    w2_bias = getattr(layer, "w2_weight_bias", None)

    for slot in range(num_topk):
        ids_slot = topk_ids[:, slot]
        weights_slot = topk_weights[:, slot].to(dtype=x.dtype)
        for start in range(0, num_tokens, _TOKEN_CHUNK):
            end = start + _TOKEN_CHUNK
            if end > num_tokens:
                end = num_tokens

            x_c = x[start:end]
            ids_c = ids_slot[start:end]
            w13_c = w13_weight[ids_c]
            w2_c = w2_weight[ids_c]

            if is_gated:
                w1, w3 = torch.chunk(w13_c, 2, dim=1)
                x1 = torch.einsum("ti,toi->to", x_c, w1)
                x3 = torch.einsum("ti,toi->to", x_c, w3)
                if w13_bias is not None:
                    b1, b3 = torch.chunk(w13_bias[ids_c], 2, dim=-1)
                    x1 = x1 + b1.to(dtype=x1.dtype)
                    x3 = x3 + b3.to(dtype=x3.dtype)
                act = _activate(x1, x3, activation)
            else:
                x_up = torch.einsum("ti,toi->to", x_c, w13_c)
                if w13_bias is not None:
                    x_up = x_up + w13_bias[ids_c].to(dtype=x_up.dtype)
                act = _activate(x_up, None, activation)

            # w2: [C, hidden, inter], act: [C, inter] -> [C, hidden]
            expert = torch.einsum("ci,chi->ch", act, w2_c)
            if w2_bias is not None:
                expert = expert + w2_bias[ids_c].to(dtype=expert.dtype)

            out[start:end] = out[start:end] + expert * weights_slot[start:end].unsqueeze(-1)

    routed_scaling = getattr(cfg, "routed_scaling_factor", None)
    if routed_scaling is not None and routed_scaling != 1.0:
        out = out * routed_scaling

    if getattr(cfg, "inplace", False):
        x.copy_(out)
        out = x

    return StandardCombineInput(hidden_states=out)
