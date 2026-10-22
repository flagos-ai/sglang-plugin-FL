# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Preserve Python -1 padding semantics in HCU speculative logprob indexing.

The empty HCU FlagGems advanced-index path does not reliably wrap negative
indices. Upstream computes both logits and predicted-token rows with the same
accept_index, whose documented -1 padding means the final row. Normalize only
that sentinel and reuse the upstream stable row normalizer. The temperature,
layout, clamp and output-helper semantics follow compute_spec_logprobs.
"""

import logging
import sys
from threading import RLock

logger = logging.getLogger(__name__)
_TARGET = "sglang.srt.layers.logprob_processor.compute_spec_logprobs"
_registration_lock = RLock()
_registered = False
_padding_logged = False
_stable_logged = False


def _is_hcu():
    # Platform discovery owns Torch import; plugin registration imports no model.
    return hasattr(sys.modules.get("torch"), "__hcu_version__")


def normalize_hcu_spec_logprob_accept_index(accept_index, num_rows):
    """Return an equivalent positive index without mutating valid token rows.

    Only -1 is an upstream padding sentinel. Other negatives and positive
    out-of-range values are errors, not values to clamp into a valid row.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        raise RuntimeError("HCU speculative logprob normalization requires Torch")
    if not isinstance(accept_index, torch.Tensor):
        raise ValueError("HCU speculative accept_index must be a tensor")
    if accept_index.ndim != 2 or accept_index.dtype not in (torch.int32, torch.int64):
        raise ValueError("HCU speculative accept_index must be a 2D integer tensor")
    if type(num_rows) is not int or num_rows < 0:
        raise ValueError("HCU speculative logits row count must be nonnegative")
    if accept_index.numel() == 0:
        return accept_index
    valid = (accept_index >= -1) & (accept_index < num_rows)
    if num_rows == 0 or not bool(valid.all().item()):
        raise ValueError("HCU speculative accept_index must lie in [-1, logits_rows)")
    padding = accept_index == -1
    if not bool(padding.any().item()):
        return accept_index
    # N-1, rather than zero, is exactly Python negative-index wrapping. In
    # particular both the chosen token and its log-softmax normalizer use it.
    return torch.where(padding, num_rows - 1, accept_index)


def _compute_spec_logprobs_hook(
    original, batch, logits_output, predict, *, accept_index=None, chain_stride=None
):
    if not _is_hcu():
        return original(
            batch,
            logits_output,
            predict,
            accept_index=accept_index,
            chain_stride=chain_stride,
        )
    assert (accept_index is None) != (chain_stride is None), (
        "pass exactly one of accept_index / chain_stride"
    )
    logits = logits_output.next_token_logits
    if logits.ndim != 2 or predict.ndim != 1 or predict.numel() != logits.shape[0]:
        raise ValueError("HCU speculative logits/predict row counts must agree")
    torch = sys.modules["torch"]
    bs = len(batch.seq_lens)
    if accept_index is not None:
        if (
            not isinstance(accept_index, torch.Tensor)
            or accept_index.ndim != 2
            or accept_index.shape[0] != bs
        ):
            raise ValueError(
                "HCU speculative accept_index rows must match the live batch"
            )
        if accept_index.device != logits.device or predict.device != logits.device:
            raise ValueError(
                "HCU speculative logits/predict/accept_index devices must agree"
            )
        normalized = normalize_hcu_spec_logprob_accept_index(
            accept_index, logits.shape[0]
        )
        if normalized is not accept_index:
            global _padding_logged
            with _registration_lock:
                if not _padding_logged:
                    logger.info(
                        "HCU MTP logprob padding normalized: shape=%s padding_count=%d last_row=%d",
                        tuple(accept_index.shape),
                        int((accept_index == -1).sum().item()),
                        logits.shape[0] - 1,
                    )
                    _padding_logged = True
        max_accept = normalized.shape[1]
        flat = normalized.long().reshape(-1)
        gathered_logits = logits[flat]
        accepted_token_ids = predict[flat]
    else:
        max_accept = chain_stride
        assert logits.shape[0] == bs * max_accept, (
            f"chain layout expects {bs * max_accept} logits rows, got {logits.shape[0]}"
        )
        gathered_logits = logits
        accepted_token_ids = predict

    # Reuse SGLang's existing row normalization and output helpers. Keep this
    # expansion in step with the v0.5.18 compute_spec_logprobs feature branches.
    from sglang.srt.layers.logprob_processor import (
        compute_row_log_normalizer,
        envs,
        get_token_ids_logprobs,
        get_top_logprobs,
    )

    if not (
        batch.sampling_info.is_all_greedy or envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get()
    ):
        temperatures = torch.repeat_interleave(
            batch.sampling_info.temperatures, max_accept, dim=0
        )
        gathered_logits = gathered_logits / temperatures
    output_dtype = gathered_logits.dtype
    row_max, shifted_log_sum = compute_row_log_normalizer(gathered_logits)
    # Two subtractions preserve small normalizers even with large row offsets.
    # Cast back to the upstream log_softmax(input) output dtype before clamping.
    logprobs = (
        (gathered_logits.float() - row_max[:, None]) - shifted_log_sum[:, None]
    ).to(output_dtype)
    logprobs.clamp_(min=torch.finfo(output_dtype).min)
    logits_output.next_token_logprobs = logprobs.gather(
        1, accepted_token_ids.long().view(-1, 1)
    ).view(bs, max_accept)
    if batch.top_logprobs_nums and any(x > 0 for x in batch.top_logprobs_nums):
        expanded = [num for num in batch.top_logprobs_nums for _ in range(max_accept)]
        (
            logits_output.next_token_top_logprobs_val,
            logits_output.next_token_top_logprobs_idx,
        ) = get_top_logprobs(logprobs, expanded, no_copy_to_cpu=True)
    if batch.token_ids_logprobs and any(
        x is not None for x in batch.token_ids_logprobs
    ):
        expanded = [ids for ids in batch.token_ids_logprobs for _ in range(max_accept)]
        (
            logits_output.next_token_token_ids_logprobs_val,
            logits_output.next_token_token_ids_logprobs_idx,
        ) = get_token_ids_logprobs(logprobs, expanded, no_copy_to_cpu=True)
    global _stable_logged
    with _registration_lock:
        if not _stable_logged:
            logger.info(
                "HCU MTP stable logprob normalization active: layout=%s rows=%d dtype=%s",
                "accept_index" if accept_index is not None else "chain_stride",
                gathered_logits.shape[0],
                output_dtype,
            )
            _stable_logged = True


def register_hcu_mtp_logprob_guard():
    """Use SGLang's public AROUND hook and its from-import alias propagation.

    Registration does not import the target. HookRegistry.apply_hooks resolves
    it after plugin loading, updating an already-loaded EAGLE common alias too;
    later natural imports receive the wrapped defining function directly.
    """
    global _registered
    if not _is_hcu():
        return False
    with _registration_lock:
        if not _registered:
            from sglang.srt.plugins.hook_registry import HookRegistry, HookType

            HookRegistry.register(_TARGET, _compute_spec_logprobs_hook, HookType.AROUND)
            _registered = True
    return True
