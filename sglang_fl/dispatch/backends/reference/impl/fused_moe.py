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

"""Reference (PyTorch) fused MoE expert computation.

Two paths:

1. **Graph-safe top-k gather** (default for decode / CUDA graph): loop over the
   fixed ``topk`` columns, gather expert weights with ``index_select`` into
   ``[T, ...]`` tensors. No ``.item()`` / ``.tolist()`` / data-dependent expert
   loop — shapes stay static across capture and replay.

2. **Eager expert-grouped** (large prefill): per-expert contiguous slices to
   avoid materializing ``[T, topk, 2I, H]``. Routing bookkeeping runs on CPU
   (Hygon device ``bincount`` / flag_gems index hangs). Not CUDA-graph safe.

Weight layout (UnquantizedFusedMoEMethod):
  w13_weight [E, 2*inter, hidden] when gated (else [E, inter, hidden])
  w2_weight  [E, hidden, inter]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Peak bytes for one top-k column of gathered w13 before falling back to the
# expert-grouped path in eager mode. Decode / CUDA-graph always uses gather.
_GATHER_W13_BYTES_LIMIT = 512 << 20  # 512 MiB


def _swiglu_gpt_oss(x: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """GPT-OSS style SwiGLU (interleaved gate/up), matches swiglu_gpt_oss_sigmoid_alpha."""
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(gate * alpha) * (up + 1)


def _swiglu_silu_clamp_mul(x: torch.Tensor, limit: float) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    gate = F.silu(gate).clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    return gate * up


def _apply_activation(
    gate_up: torch.Tensor,
    *,
    activation: str,
    is_gated: bool,
    gemm1_alpha: float | None,
    gemm1_limit: float | None,
) -> torch.Tensor:
    if is_gated:
        if activation == "silu" and gemm1_alpha is not None:
            if gemm1_limit is None:
                raise ValueError("gemm1_clamp_limit required when gemm1_alpha is set")
            return _swiglu_gpt_oss(gate_up, gemm1_alpha, gemm1_limit)
        if activation == "silu" and gemm1_limit is not None:
            return _swiglu_silu_clamp_mul(gate_up, gemm1_limit)
        if activation == "silu":
            d = gate_up.shape[-1] // 2
            return F.silu(gate_up[..., :d]) * gate_up[..., d:]
        if activation == "gelu":
            d = gate_up.shape[-1] // 2
            return F.gelu(gate_up[..., :d]) * gate_up[..., d:]
        raise ValueError(f"Unsupported gated activation: {activation=}")

    if activation == "silu":
        return F.silu(gate_up)
    if activation == "gelu":
        return F.gelu(gate_up)
    if activation == "relu2":
        return torch.square(F.relu(gate_up))
    raise ValueError(f"Unsupported activation: {activation=}, is_gated=False")


def _index_select(x: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """ATen index_select — bypasses flag_gems Tensor.__getitem__ / index patches."""
    return torch.ops.aten.index_select(x, dim, index.contiguous())


def _index_copy(
    dst: torch.Tensor, dim: int, index: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    """ATen index_copy — bypasses flag_gems advanced-index store patches."""
    return torch.ops.aten.index_copy(dst, dim, index.contiguous(), source)


def _is_stream_capturing(device: torch.device) -> bool:
    """True while a CUDA/HIP graph is being captured on the current stream."""
    if device.type not in ("cuda", "hip"):
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _prefer_graph_safe_gather(
    *,
    num_tokens: int,
    inter: int,
    hidden: int,
    elem_size: int,
    device: torch.device,
) -> bool:
    if _is_stream_capturing(device):
        return True
    # One top-k column of gathered w13: [T, inter, H]
    est = num_tokens * inter * hidden * elem_size
    return est <= _GATHER_W13_BYTES_LIMIT


def _moe_topk_gather(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer: torch.nn.Module,
    cfg,
) -> torch.Tensor:
    """Fixed-shape per-topk-column gather — CUDA-graph / torch.compile friendly."""
    num_tokens, hidden = x.shape
    topk = topk_ids.shape[1]
    w13_bias = getattr(layer, "w13_weight_bias", None)
    w2_bias = getattr(layer, "w2_weight_bias", None)

    out = torch.zeros(num_tokens, hidden, device=x.device, dtype=x.dtype)
    # topk is a Python int from tensor shape — static across capture/replay.
    for k in range(topk):
        ids_k = topk_ids[:, k]
        valid = ids_k >= 0
        safe_ids = ids_k.clamp(min=0)

        w13 = _index_select(layer.w13_weight, 0, safe_ids)  # [T, 2I|I, H]
        gate_up = torch.bmm(x.unsqueeze(1), w13.transpose(1, 2)).squeeze(1)
        if w13_bias is not None:
            bias13 = _index_select(w13_bias, 0, safe_ids)
            gate_up = (gate_up.float() + bias13).to(dtype=x.dtype)

        mid = _apply_activation(
            gate_up,
            activation=cfg.activation,
            is_gated=cfg.is_gated,
            gemm1_alpha=cfg.gemm1_alpha,
            gemm1_limit=cfg.gemm1_clamp_limit,
        )

        w2 = _index_select(layer.w2_weight, 0, safe_ids)  # [T, H, I]
        expert_out = torch.bmm(mid.unsqueeze(1), w2.transpose(1, 2)).squeeze(1)
        if w2_bias is not None:
            bias2 = _index_select(w2_bias, 0, safe_ids)
            expert_out = (expert_out.float() + bias2).to(dtype=x.dtype)

        weight = topk_weights[:, k].to(dtype=expert_out.dtype).unsqueeze(-1)
        weight = weight * valid.to(device=x.device, dtype=weight.dtype).unsqueeze(-1)
        out = out + expert_out * weight

    if cfg.routed_scaling_factor is not None and cfg.routed_scaling_factor != 1.0:
        out = out * cfg.routed_scaling_factor
    return out


def _moe_expert_grouped(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer: torch.nn.Module,
    cfg,
) -> torch.Tensor:
    """Per-expert contiguous GEMM — lower peak memory for large eager prefill."""
    num_tokens, hidden = x.shape
    topk = topk_ids.shape[1]
    num_experts = layer.w13_weight.shape[0]

    # CPU routing only (ids); keep gather / GEMM on GPU.
    flat_ids_cpu = topk_ids.reshape(-1).detach().to(dtype=torch.int64, device="cpu")
    work_ids_cpu = flat_ids_cpu.clone()
    work_ids_cpu.masked_fill_(work_ids_cpu < 0, num_experts)
    idxs_cpu = work_ids_cpu.argsort()
    tokens_per_expert = torch.bincount(work_ids_cpu, minlength=num_experts + 1)[
        :num_experts
    ]
    tokens_per_expert_list = tokens_per_expert.tolist()
    n_valid = int(tokens_per_expert.sum().item())

    idxs = idxs_cpu.to(device=x.device, non_blocking=False)
    token_rows = torch.div(idxs, topk, rounding_mode="floor")
    sorted_tokens = _index_select(x, 0, token_rows)

    w13_bias = getattr(layer, "w13_weight_bias", None)
    w2_bias = getattr(layer, "w2_weight_bias", None)

    outputs = []
    start_idx = 0
    for expert_id, n in enumerate(tokens_per_expert_list):
        end_idx = start_idx + n
        if n == 0:
            continue

        tokens_e = sorted_tokens.narrow(0, start_idx, n)
        original_dtype = tokens_e.dtype

        gate_up = F.linear(tokens_e, layer.w13_weight[expert_id])
        if w13_bias is not None:
            gate_up = (gate_up.float() + w13_bias[expert_id]).to(original_dtype)

        mid = _apply_activation(
            gate_up,
            activation=cfg.activation,
            is_gated=cfg.is_gated,
            gemm1_alpha=cfg.gemm1_alpha,
            gemm1_limit=cfg.gemm1_clamp_limit,
        )

        expert_out = F.linear(mid, layer.w2_weight[expert_id])
        if w2_bias is not None:
            expert_out = (expert_out.float() + w2_bias[expert_id]).to(original_dtype)

        outputs.append(expert_out)
        start_idx = end_idx

    flat_out = torch.zeros(num_tokens * topk, hidden, device=x.device, dtype=x.dtype)
    if outputs and n_valid > 0:
        outs = torch.cat(outputs, dim=0)
        flat_out = _index_copy(flat_out, 0, idxs.narrow(0, 0, n_valid), outs)

    out = (
        flat_out.view(num_tokens, topk, hidden)
        .to(dtype=topk_weights.dtype)
        .mul_(topk_weights.unsqueeze(-1))
        .sum(dim=1)
        .to(dtype=x.dtype)
    )

    if cfg.routed_scaling_factor is not None and cfg.routed_scaling_factor != 1.0:
        out = out * cfg.routed_scaling_factor
    return out


def fused_moe_torch(
    obj,
    layer: torch.nn.Module,
    dispatch_output,
):
    """
    Pure-PyTorch fused MoE for the reference backend.

    Prefers the fixed-shape top-k gather path (CUDA-graph safe). Falls back to
    the expert-grouped path only for large eager prefills where gather would OOM.

    Args:
        obj: UnquantizedFusedMoEMethod (provides ``moe_runner_config``)
        layer: MoE layer with ``w13_weight`` / ``w2_weight`` (and optional biases)
        dispatch_output: StandardDispatchOutput (hidden_states, topk_output)

    Returns:
        StandardCombineInput(hidden_states=...)
    """
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    x = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    topk_weights, topk_ids, _ = topk_output

    cfg = getattr(obj, "moe_runner_config", None) or getattr(
        layer, "moe_runner_config", None
    )
    if cfg is None:
        raise RuntimeError(
            "fused_moe reference requires moe_runner_config on obj or layer"
        )

    if cfg.apply_router_weight_on_input:
        raise NotImplementedError(
            "reference fused_moe does not support apply_router_weight_on_input"
        )

    if x.numel() == 0:
        return StandardCombineInput(hidden_states=x)

    num_tokens, hidden = x.shape
    inter = layer.w13_weight.shape[1]

    if _prefer_graph_safe_gather(
        num_tokens=num_tokens,
        inter=inter,
        hidden=hidden,
        elem_size=x.element_size(),
        device=x.device,
    ):
        out = _moe_topk_gather(x, topk_weights, topk_ids, layer, cfg)
    else:
        out = _moe_expert_grouped(x, topk_weights, topk_ids, layer, cfg)

    if cfg.inplace:
        x.copy_(out)
        return StandardCombineInput(hidden_states=x)

    return StandardCombineInput(hidden_states=out)
