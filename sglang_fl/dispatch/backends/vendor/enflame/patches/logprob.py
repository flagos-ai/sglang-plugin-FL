"""GCU logprob top-k compatibility; model inference stays on GCU."""

from __future__ import annotations

import torch


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def top_logprobs_raw_gcu(
    original,
    logprobs,
    top_logprobs_nums,
    stage,
    extend_logprob_pruned_lens_cpu=None,
    no_copy_to_cpu=False,
):
    args = (top_logprobs_nums, stage, extend_logprob_pruned_lens_cpu, no_copy_to_cpu)
    if logprobs.device.type != "gcu":
        return original(logprobs, *args)
    # The installed GCU native top-k requires int32 indices, while ATen requires
    # int64. CPU selection preserves the public index dtype and exact values.
    result = original(logprobs.cpu(), *args)
    return _to_device(result, logprobs.device) if no_copy_to_cpu else result


def top_logprobs_chunk_gcu(
    original,
    logprobs,
    top_k_nums,
    pruned_lens,
    top_logprobs_val,
    top_logprobs_idx,
    split_pruned_len,
    log_normalizer=None,
    precomputed_topk=None,
):
    if logprobs.device.type == "gcu":
        logprobs = logprobs.cpu()
        if log_normalizer is not None:
            log_normalizer = tuple(value.cpu() for value in log_normalizer)
        if precomputed_topk is not None:
            precomputed_topk = tuple(value.cpu() for value in precomputed_topk)
    return original(
        logprobs,
        top_k_nums,
        pruned_lens,
        top_logprobs_val,
        top_logprobs_idx,
        split_pruned_len,
        log_normalizer=log_normalizer,
        precomputed_topk=precomputed_topk,
    )


def patch_logprob():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    for name, hook in (
        ("get_top_logprobs_raw", top_logprobs_raw_gcu),
        ("get_top_logprobs_chunk", top_logprobs_chunk_gcu),
    ):
        HookRegistry.register(
            f"sglang.srt.layers.logprob_processor.{name}", hook, HookType.AROUND
        )
