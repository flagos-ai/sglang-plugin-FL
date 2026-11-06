"""GCU routing for the SGLang EAGLE tree and cache-location APIs."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional

import torch
import triton

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_utils import TreeMaskMode


def build_tree_gcu(original, bonus_tokens, *args, **kwargs):
    if bonus_tokens.device.type != "gcu":
        return original(bonus_tokens, *args, **kwargs)
    return _build_tree_gcu(bonus_tokens, *args, **kwargs)


def verify_greedy_gcu(
    original,
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrieve_index,
    retrieve_next_token,
    retrieve_next_sibling,
    target_predict,
    topk=-1,
):
    values = dict(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrieve_index=retrieve_index,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        target_predict=target_predict,
    )
    if predicts.device.type != "gcu":
        return original(**values, topk=topk)
    from sglang.srt.speculative.eagle_utils import verify_tree_greedy_triton

    verify_tree_greedy_triton(**values)
    return predicts, accept_index, accept_token_num


def assign_uniform_gcu(
    original,
    req_pool_indices,
    req_to_token,
    start_offset,
    batch_size,
    draft_token_num,
    device,
):
    if req_pool_indices.device.type != "gcu":
        return original(
            req_pool_indices, req_to_token, start_offset,
            batch_size, draft_token_num, device,
        )
    from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_uniform

    locs = torch.empty(batch_size * draft_token_num, dtype=torch.int64, device=device)
    if batch_size:
        assign_extend_cache_locs_uniform[(batch_size,)](
            req_pool_indices, req_to_token, start_offset, locs,
            req_to_token.shape[1], draft_token_num,
        )
    return locs


def assign_cache_locs_gcu(
    original,
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    batch_size,
    draft_token_num,
    device,
):
    if req_pool_indices.device.type != "gcu":
        return original(
            req_pool_indices, req_to_token, start_offset, end_offset,
            batch_size, draft_token_num, device,
        )
    from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs

    locs = torch.empty(batch_size * draft_token_num, dtype=torch.int64, device=device)
    if batch_size:
        assign_extend_cache_locs[(batch_size,)](
            req_pool_indices, req_to_token, start_offset, end_offset, locs,
            req_to_token.shape[1], triton.next_power_of_2(batch_size),
        )
    return locs


def gdn_verify_gcu(original, *args, **kwargs):
    q = kwargs.get("q") if "q" in kwargs else args[5]
    if q.device.type != "gcu":
        return original(*args, **kwargs)
    from ..impl.gdn_mtp_verify import fused_sigmoid_gating_delta_rule_update_gcu

    return fused_sigmoid_gating_delta_rule_update_gcu(*args, **kwargs)


def patch_speculative():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    targets = (
        ("sglang.srt.speculative.eagle_utils.build_tree_kernel_efficient", build_tree_gcu),
        ("sglang.srt.speculative.eagle_utils.verify_tree_greedy_func", verify_greedy_gcu),
        ("sglang.kernels.ops.speculative.cache_locs.assign_extend_cache_locs_uniform_func", assign_uniform_gcu),
        ("sglang.kernels.ops.speculative.cache_locs.assign_extend_cache_locs_func", assign_cache_locs_gcu),
        ("sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent.fused_sigmoid_gating_delta_rule_update", gdn_verify_gcu),
    )
    for target, hook in targets:
        HookRegistry.register(target, hook, HookType.AROUND)


# Tree construction adapted from SGLang v0.5.18 (Apache-2.0).
def _build_tree_gcu(
    bonus_tokens: torch.Tensor,
    parent_list: List[torch.Tensor],
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: TreeMaskMode = 0,
    tree_mask_buf: Optional[torch.Tensor] = None,
    fill_prefix_mask: bool = True,
):
    from sglang.srt.speculative.eagle_utils import (
        TreeMaskMode, sgl_build_tree_kernel_triton,
    )

    draft_tokens = torch.cat((bonus_tokens.unsqueeze(1), draft_tokens), dim=1).flatten()

    # seq_lens_sum == sum(seq_lens); seq_lens: sequence length without draft tokens
    bs = seq_lens.numel()
    device = seq_lens.device
    # e.g. for bs=1, tree_mask: num_draft_token, seq_lens_sum + num_draft_token (flattened)
    # where each row indicates the attending pattern of each draft token
    # if use_partial_packed_tree_mask is True, tree_mask: num_draft_token (flattened, packed)
    if tree_mask_buf is not None:
        tree_mask = tree_mask_buf
        if tree_mask_mode == TreeMaskMode.QLEN_ONLY:
            tree_mask.fill_(True)
        elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
            tree_mask.fill_(0)
        elif tree_mask_mode == TreeMaskMode.FULL_MASK:
            # Only the [0, seq_len) prefix columns depend on this fill; the
            # kernel below writes every tree cell itself. Skip the (up to
            # 100s of MB) per-step memset when nothing reads the mask.
            if fill_prefix_mask:
                tree_mask.fill_(True)
        else:
            raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY:
        tree_mask = torch.full(
            (num_verify_tokens * bs * num_verify_tokens,),
            True,
            dtype=torch.bool,
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        packed_dtypes = [torch.uint8, torch.uint16, torch.uint32]
        packed_dtype_idx = int(math.ceil(math.log2((num_verify_tokens + 7) // 8)))
        tree_mask = torch.zeros(
            (num_verify_tokens * bs,),
            dtype=packed_dtypes[packed_dtype_idx],
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.FULL_MASK:
        mask_shape = (
            seq_lens_sum * num_verify_tokens
            + num_verify_tokens * num_verify_tokens * bs,
        )
        # Same reasoning as the preallocated branch above.
        tree_mask = (
            torch.full(mask_shape, True, dtype=torch.bool, device=device)
            if fill_prefix_mask
            else torch.empty(mask_shape, dtype=torch.bool, device=device)
        )
    else:
        raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")

    # TODO: make them torch.empty and fuse them into `sgl_build_tree_kernel`
    retrieve_buf = torch.full(
        (3, bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrieve_index, retrieve_next_token, retrieve_next_sibling = retrieve_buf
    # position: where each token belongs to
    # e.g. if depth of each draft token is [0, 1, 1, 2] and the prompt length is 7
    # then, positions = [7, 8, 8, 9]
    positions = torch.empty((bs * num_verify_tokens,), device=device, dtype=torch.long)

    sgl_build_tree_kernel_triton(
        parent_list, top_scores_index, seq_lens, tree_mask, positions,
        retrieve_index, retrieve_next_token, retrieve_next_sibling,
        topk, spec_steps, num_verify_tokens, tree_mask_mode,
    )
    return (
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        draft_tokens,
    )
