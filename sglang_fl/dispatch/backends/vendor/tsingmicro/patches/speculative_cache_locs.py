# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Speculative-decoding KV slot bookkeeping on txda (SGLang 0.5.18).

Three helpers that build or record KV write locations are wrong on txda:

* ``sglang.kernels.ops.speculative.cache_locs.assign_extend_cache_locs_func`` /
  ``assign_extend_cache_locs_uniform_func`` (verify ``out_cache_loc`` in
  ``eagle_prepare_for_verify``) have cuda/hip/musa/xpu, npu and cpu branches
  only and return ``None`` on txda; the verify forward then fails writing KV.
* The ``assign_extend_cache_locs`` Triton kernel itself writes nothing on txda
  (one-card check vs torch: output left untouched). ``spec_utils`` launches it
  directly in ``move_accept_tokens_to_target_kvcache`` (tree verify, topk > 1).
* ``sglang.srt.mem_cache.allocation.assign_req_to_token_pool_func`` (draft KV
  slots -> ``req_to_token`` in ``alloc_for_spec_decode``): its Triton kernel
  writes only some of the slots on txda. Unwritten ``req_to_token`` entries stay
  stale/zero, so attention reads the wrong KV (low acceptance, garbage output)
  and the release path finds "phantom zero entries" (KV pool leak check).

All three only gather/scatter row slices of ``req_to_token``; on txda they are
done with torch indexing. The module attributes are replaced right after SGLang
imports each module, before ``spec_utils`` / ``eagle_utils`` bind the names.
"""

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger
from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._utils import (
    _install_post_import_wrap,
)

_log = patch_logger("speculative_cache_locs")

_MODULE = "sglang.kernels.ops.speculative.cache_locs"
_patched = False


def assign_extend_cache_locs_func(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    batch_size: int,
    draft_token_num: int,
    device,
):
    """Same contract as upstream: rows' [start, end) slices packed back to back
    into a ``batch_size * draft_token_num`` int64 tensor."""
    import torch

    out = torch.empty((batch_size * draft_token_num,), dtype=torch.int64, device=device)
    start = start_offset[:batch_size].to(torch.int64)
    lens = end_offset[:batch_size].to(torch.int64) - start
    total = int(lens.sum().item())
    if total == 0:
        return out
    rows = torch.repeat_interleave(torch.arange(batch_size, device=start.device), lens)
    row_begin = torch.cumsum(lens, 0) - lens  # output offset of each row
    pos = start[rows] + (torch.arange(total, device=start.device) - row_begin[rows])
    req = req_pool_indices[:batch_size].to(torch.int64)[rows]
    out[:total] = req_to_token[req, pos].to(torch.int64)
    return out


def assign_extend_cache_locs_uniform_func(
    req_pool_indices,
    req_to_token,
    start_offset,
    batch_size: int,
    draft_token_num: int,
    device,
):
    """Uniform case: every row extends exactly ``draft_token_num`` tokens."""
    import torch

    req = req_pool_indices[:batch_size].to(torch.int64)[:, None]
    pos = start_offset[:batch_size].to(torch.int64)[:, None] + torch.arange(
        draft_token_num, device=req.device
    )[None, :]
    return req_to_token[req, pos].reshape(-1).to(device=device, dtype=torch.int64)


def assign_req_to_token_pool_func(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    batch_size: int,
):
    """Write ``out_cache_loc`` (packed rows) into ``req_to_token[req, start:end]``."""
    import torch

    start = start_offset[:batch_size].to(torch.int64)
    lens = end_offset[:batch_size].to(torch.int64) - start
    total = int(lens.sum().item())
    if total == 0:
        return
    rows = torch.repeat_interleave(torch.arange(batch_size, device=start.device), lens)
    row_begin = torch.cumsum(lens, 0) - lens
    pos = start[rows] + (torch.arange(total, device=start.device) - row_begin[rows])
    req = req_pool_indices[:batch_size].to(torch.int64)[rows]
    req_to_token[req, pos] = out_cache_loc[:total].to(req_to_token.dtype)


class _AssignExtendCacheLocsLaunch:
    """Stand-in for the ``assign_extend_cache_locs`` Triton kernel object:
    ``kernel[(bs,)](req_pool_indices, req_to_token, start, end, out, pool_len, bs_upper)``."""

    _txda = True

    def __getitem__(self, grid):
        def launch(req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc, *_):
            bs = grid[0]
            n = out_cache_loc.shape[0]
            res = assign_extend_cache_locs_func(
                req_pool_indices, req_to_token, start_offset, end_offset,
                bs, max(1, n // max(1, bs)), out_cache_loc.device,
            )
            k = min(n, res.shape[0])
            out_cache_loc[:k] = res[:k].to(out_cache_loc.dtype)

        return launch


def _apply_allocation() -> None:
    try:
        import sglang.srt.mem_cache.allocation as alloc

        if getattr(alloc.assign_req_to_token_pool_func, "_txda", False):
            return
        assign_req_to_token_pool_func._txda = True
        alloc.assign_req_to_token_pool_func = assign_req_to_token_pool_func
        _log.applied("allocation.assign_req_to_token_pool_func -> torch scatter (txda)")
    except Exception as exc:  # noqa: BLE001
        _log.failed("cannot patch allocation.assign_req_to_token_pool_func: %s", exc)


def _apply_spec_utils() -> None:
    try:
        import sglang.srt.speculative.spec_utils as su

        su.assign_extend_cache_locs = _AssignExtendCacheLocsLaunch()
        if not getattr(su.assign_req_to_token_pool_func, "_txda", False):
            su.assign_req_to_token_pool_func = assign_req_to_token_pool_func
        _log.applied("spec_utils assign_extend_cache_locs / assign_req_to_token_pool -> torch (txda)")
    except Exception as exc:  # noqa: BLE001
        _log.failed("cannot patch spec_utils cache-loc helpers: %s", exc)


def _apply() -> None:
    try:
        import sglang.kernels.ops.speculative.cache_locs as cl

        if getattr(cl.assign_extend_cache_locs_func, "_txda", False):
            return
        assign_extend_cache_locs_func._txda = True
        assign_extend_cache_locs_uniform_func._txda = True
        cl.assign_extend_cache_locs_func = assign_extend_cache_locs_func
        cl.assign_extend_cache_locs_uniform_func = assign_extend_cache_locs_uniform_func
        _log.applied("speculative assign_extend_cache_locs(_uniform)_func -> torch gather (txda)")
    except Exception as exc:  # noqa: BLE001
        _log.failed("cannot patch speculative cache_locs: %s", exc)


def patch() -> None:
    """Install the lazy patch. Idempotent. txda-only (called from the txda block)."""
    global _patched
    if _patched:
        return
    _install_post_import_wrap(_MODULE, _apply)
    _install_post_import_wrap("sglang.srt.mem_cache.allocation", _apply_allocation)
    _install_post_import_wrap("sglang.srt.speculative.spec_utils", _apply_spec_utils)
    _patched = True
