# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


def is_available() -> bool:
    try:
        import ixformer.inference.functions as ops
    except ImportError:
        return False
    return all(
        hasattr(ops, name)
        for name in (
            "moe_compute_token_index",
            "moe_expand_input",
            "moe_w16a16_group_gemm",
            "moe_output_reduce_sum",
        )
    )


def fused_experts(
    hidden: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    import ixformer.inference.functions as ops

    tokens, hidden_size = hidden.shape
    topk = topk_ids.shape[1]
    experts = w13.shape[0]
    ids = topk_ids.to(torch.int32).contiguous()
    source_to_dest, dest_to_source, sizes_gpu, sizes_cpu = (
        ops.moe_compute_token_index(ids, experts)
    )
    if sizes_cpu is None:
        sizes_cpu = sizes_gpu.cpu()
    sizes_cpu = sizes_cpu.to(torch.int32)
    grouped = ops.moe_expand_input(
        hidden.contiguous(),
        dest_to_source,
        tokens * topk,
        topk,
        src_to_dst=source_to_dest,
    )
    gate_up = ops.moe_w16a16_group_gemm(
        grouped,
        w13.contiguous(),
        output_dtype=hidden.dtype,
        tokens_per_experts=sizes_cpu,
        format="TN",
    )

    from .triton_ops import silu_and_mul

    activated = silu_and_mul(gate_up)
    if activated is None:
        gate, up = gate_up.chunk(2, dim=-1)
        activated = torch.nn.functional.silu(gate) * up
    expert_output = ops.moe_w16a16_group_gemm(
        activated,
        w2.contiguous(),
        output_dtype=hidden.dtype,
        tokens_per_experts=sizes_cpu,
        format="TN",
    )
    token_major = expert_output.index_select(0, source_to_dest.long())
    return ops.moe_output_reduce_sum(
        token_major.view(tokens, topk, hidden_size),
        topk_weight=topk_weights.float().contiguous(),
    )
