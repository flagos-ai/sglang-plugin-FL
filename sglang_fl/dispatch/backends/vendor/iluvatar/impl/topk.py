from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch


@lru_cache(maxsize=1)
def _router_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        logits,
        correction,
        weights,
        ids,
        experts: tl.constexpr,
        TOPK: tl.constexpr,
        GROUPS: tl.constexpr,
        TOP_GROUPS: tl.constexpr,
        GROUPED: tl.constexpr,
        SIGMOID: tl.constexpr,
        HAS_CORRECTION: tl.constexpr,
        RENORMALIZE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        mask = col < experts
        values = tl.load(logits + row * experts + col, mask=mask, other=-float("inf"))
        if SIGMOID:
            score = tl.where(mask, tl.sigmoid(values.to(tl.float32)), -float("inf"))
        else:
            score = tl.where(mask, tl.softmax(values.to(tl.float32)), -float("inf"))
        allowed = mask
        if GROUPED:
            EPG: tl.constexpr = experts // GROUPS
            group_score = tl.max(tl.reshape(score, (GROUPS, EPG)), axis=1)
            selected_groups = tl.zeros((GROUPS,), dtype=tl.int1)
            selected_experts = tl.zeros((BLOCK,), dtype=tl.int1)
            for _ in tl.static_range(TOP_GROUPS):
                group = tl.argmax(
                    tl.where(selected_groups, -float("inf"), group_score), axis=0
                )
                selected_groups |= tl.arange(0, GROUPS) == group
                selected_experts |= (col // EPG) == group
            allowed &= selected_experts
        work = tl.where(allowed, score, -float("inf"))
        total = 0.0
        for i in tl.static_range(TOPK):
            ranked = work
            if HAS_CORRECTION:
                ranked += tl.load(correction + col, mask=mask, other=0.0)
            index = tl.argmax(ranked, axis=0)
            value = tl.sum(tl.where(col == index, score, 0.0), axis=0)
            tl.store(ids + row * TOPK + i, index)
            tl.store(weights + row * TOPK + i, value)
            total += value
            work = tl.where(col == index, -float("inf"), work)
        if RENORMALIZE:
            for i in tl.static_range(TOPK):
                ptr = weights + row * TOPK + i
                tl.store(ptr, tl.load(ptr) / total)

    return kernel


def _route_triton(logits: torch.Tensor, cfg):
    if not logits.is_cuda or logits.dim() != 2:
        return None

    import triton

    tokens, experts = logits.shape
    block = triton.next_power_of_2(experts)
    if block > 1024:
        return None
    topk = cfg.top_k
    correction = cfg.correction_bias
    weights = torch.empty((tokens, topk), device=logits.device, dtype=torch.float32)
    ids = torch.empty((tokens, topk), device=logits.device, dtype=torch.int32)
    values = logits.float().contiguous()
    correction_arg = (
        correction.float().contiguous() if correction is not None else values
    )
    _router_kernel()[(tokens,)](
        values,
        correction_arg,
        weights,
        ids,
        experts=experts,
        TOPK=topk,
        GROUPS=cfg.num_expert_group or 1,
        TOP_GROUPS=cfg.topk_group or 1,
        GROUPED=cfg.use_grouped_topk,
        SIGMOID=cfg.scoring_func == "sigmoid",
        HAS_CORRECTION=correction is not None,
        RENORMALIZE=cfg.renormalize,
        BLOCK=block,
        num_warps=2 if tokens == 1 else 1,
        num_stages=1,
    )
    return weights, ids


def _route_ixformer(logits: torch.Tensor, cfg):
    if (
        cfg.use_grouped_topk
        or cfg.correction_bias is not None
        or cfg.scoring_func != "softmax"
    ):
        return None
    try:
        from ixformer.inference.functions import moe_topk_softmax
    except ImportError:
        return None

    weights = torch.empty(
        (logits.shape[0], cfg.top_k), device=logits.device, dtype=torch.float32
    )
    ids = torch.empty(
        (logits.shape[0], cfg.top_k), device=logits.device, dtype=torch.int32
    )
    try:
        moe_topk_softmax(
            logits.float().contiguous(),
            cfg.top_k,
            weights,
            ids,
            cfg.renormalize,
        )
    except (RuntimeError, TypeError, ValueError):
        return None
    return weights, ids


def _route_torch(logits: torch.Tensor, cfg):
    score = (
        torch.sigmoid(logits.float())
        if cfg.scoring_func == "sigmoid"
        else torch.softmax(logits.float(), dim=-1)
    )
    if cfg.use_grouped_topk:
        groups = cfg.num_expert_group
        experts_per_group = score.shape[-1] // groups
        group_score = score.view(score.shape[0], groups, experts_per_group).amax(-1)
        group_ids = group_score.topk(cfg.topk_group, dim=-1).indices
        allowed = torch.zeros_like(group_score, dtype=torch.bool)
        allowed.scatter_(1, group_ids, True)
        score = score.masked_fill(
            ~allowed.unsqueeze(-1)
            .expand(-1, -1, experts_per_group)
            .reshape_as(score),
            float("-inf"),
        )
    ranked = score
    if cfg.correction_bias is not None:
        ranked = ranked + cfg.correction_bias.float()
    ids = ranked.topk(cfg.top_k, dim=-1).indices
    weights = score.gather(1, ids)
    if cfg.renormalize:
        weights /= weights.sum(-1, keepdim=True)
    return weights, ids.to(torch.int32)


def topk_iluvatar(
    obj,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info=None,
):
    cfg = obj.topk_config
    if (
        cfg.custom_routing_function is not None
        or cfg.num_fused_shared_experts
        or cfg.scoring_func not in ("softmax", "sigmoid")
        or num_token_non_padded is not None
        or expert_location_dispatch_info is not None
    ):
        raise NotImplementedError("unsupported Iluvatar TopK configuration")

    routed = _route_ixformer(router_logits, cfg)
    if routed is None:
        routed = _route_triton(router_logits, cfg)
    weights, ids = routed if routed is not None else _route_torch(router_logits, cfg)
    if cfg.apply_routed_scaling_factor_on_output and cfg.routed_scaling_factor:
        weights *= float(cfg.routed_scaling_factor)

    from sglang.srt.layers.moe.topk import StandardTopKOutput

    output = StandardTopKOutput(
        topk_weights=weights,
        topk_ids=ids,
        router_logits=router_logits,
    )
    return obj._apply_waterfill(output, hidden_states.shape[0])
