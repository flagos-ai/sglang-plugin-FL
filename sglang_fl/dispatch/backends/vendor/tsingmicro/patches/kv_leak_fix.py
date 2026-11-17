# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardened KV-page release for the sglang ``token_to_kv_pool`` leak (v3 -> v4).

On txda, multi-chunk prefill requests orphan KV pages and permanently drain the
pool.  On by default on txda; set ``SGLANG_KVLEAK_FIX=0`` to disable.  Each
helper below documents the part of the fix it implements.  SGLang 0.5.18.
"""

import logging
import os
import sys

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger
from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._utils import (
    _install_post_import_wrap,
)

logger = logging.getLogger(__name__)
_log = patch_logger("kv_leak_fix")

_patched = False
_ENABLED_KEY = "SGLANG_KVLEAK_FIX"
# write_cache_indices is defined in sglang.srt.mem_cache.allocation and called
# from alloc_for_extend in the same module, so that module's global is the one
# to replace.  Replaced lazily once sglang itself imports the module, to avoid
# pulling the serving stack at import time (the v3 lesson).
_WRITE_INDICES_MODULE = "sglang.srt.mem_cache.allocation"


def _enabled() -> bool:
    v = os.getenv(_ENABLED_KEY, "1")
    return v.lower() in ("1", "true", "yes", "on")


def _owned_pages(self, req) -> int:
    """Return how many KV pages this request's current incarnation provably owns,
    or -1 when it is unsafe to repair (stale tail in a reused slot)."""
    row = self.req_to_token_pool.req_to_token[req.req_pool_idx]
    nnz = row > 0
    if not nnz.any():
        return 0  # no pages actually allocated — nothing to repair, normal no-op

    bound = len(req.origin_input_ids) + len(req.output_ids)
    if bound <= 0:
        _log.warning("skip repair rid=%s abnormal bound=%d (no prompt?)", req.rid, bound)
        return -1

    if (row[bound:] > 0).any():
        # Slot reused and the row tail holds the previous request's stale pages
        # (already in the free table); freeing again would double-free — keep
        # the leak rather than risk it.
        _log.warning(
            "skip repair rid=%s unsafe stale-tail beyond bound=%d (row_len=%d)",
            req.rid, bound, row.shape[0],
        )
        return -1

    return int(nnz[:bound].sum().item())


# ─── 0) Root cause (v4): force write_cache_indices onto the torch fallback, bypass the broken txda Triton path ───


def _apply_write_indices_wrap() -> None:
    """Replace allocation.write_cache_indices with the non-Triton torch fallback.

    Root cause: on Triton attention backends, ``write_req_to_token_pool_triton``
    writes the prefill indices; its prefix-write segment reloads a uint64
    ``data_ptr`` and reinterprets it as an int64 pointer, which silently fails
    on txda — zeroing earlier chunks' already-written rows on multi-chunk
    requests and orphaning their pages.  The torch fallback loop (the path
    non-triton backends already use) persists every chunk correctly, at the
    cost of a few ``.item()`` CPU syncs per prefill.

    Idempotent.  Applied lazily via _install_post_import_wrap — not at
    patch() time — to avoid importing the serving stack too early.
    """
    target = sys.modules[_WRITE_INDICES_MODULE]
    if getattr(target.write_cache_indices, "_kvleak_fallback", False):
        return

    def _fallback_write_cache_indices(
        out_cache_loc,
        req_pool_indices_tensor,
        req_pool_indices_cpu,
        prefix_lens_tensor,
        prefix_lens_cpu,
        seq_lens_tensor,
        seq_lens_cpu,
        extend_lens_tensor,
        extend_lens_cpu,
        prefix_tensors,
        req_to_token_pool,
    ):
        # Same as the non-Triton branch of upstream write_cache_indices.
        pt = 0
        for i in range(req_pool_indices_cpu.shape[0]):
            req_idx = req_pool_indices_cpu[i].item()
            prefix_len = prefix_lens_cpu[i].item()
            seq_len = seq_lens_cpu[i].item()
            extend_len = extend_lens_cpu[i].item()

            req_to_token_pool.write(
                (req_idx, slice(0, prefix_len)),
                prefix_tensors[i],
            )
            req_to_token_pool.write(
                (req_idx, slice(prefix_len, seq_len)),
                out_cache_loc[pt : pt + extend_len],
            )
            pt += extend_len

    _fallback_write_cache_indices._kvleak_fallback = True
    target.write_cache_indices = _fallback_write_cache_indices
    _log.applied(
        "forced non-Triton %s.write_cache_indices (txda Triton prefix-write bypass)",
        target.__name__,
    )


def _safe_write_indices_wrap() -> None:
    """Post-import callback: never let a patch failure break sglang's own import."""
    try:
        _apply_write_indices_wrap()
    except Exception as exc:  # noqa: BLE001
        _log.failed("write_cache_indices wrap failed: %s", exc)


# ─── 2) Anti-pollution + safety net: rewrite the ChunkCache.cache_finished_req release segment ───


def _safe_cache_finished_req(self, req, is_insert=True, *, kv_len_to_handle):
    """Release-side hardening (retained from v2/v3).

    (1) Free only real pages (index > 0): a 0 is an "accounted-as-allocated
    but never written" hole that must not enter the free table — 0 is a
    reserved slot, and freeing it would let a future alloc double-assign it.
    (2) REPAIRED safety net: when commit=0/alloc=0 but the row still holds
    pages (the reset_for_retract zeroing path), free them so the leak closes
    instead of growing.
    """
    # Same range as upstream: release_kv_cache() passes the length; the
    # protected prefix [0, cache_protected_len) is not this req's to free, and
    # the over-allocated tail is released by the caller afterwards.
    row = self.req_to_token_pool.req_to_token[req.req_pool_idx]
    kv_indices = row[req.cache_protected_len : kv_len_to_handle]
    kv_committed_len = int(kv_indices.shape[0])
    # Page indices start at 1 (allocator's free_pages = arange(1, size+1)),
    # so the >0 test is exactly "real page" — no false hits.
    mask = kv_indices > 0
    real = int(mask.sum().item())
    if real > 0:
        self.token_to_kv_pool_allocator.free(kv_indices[mask])
        if real != kv_committed_len:
            _log.warning(
                "FILTERED %d phantom zero entries on release (rid=%s) commit=%d real=%d",
                kv_committed_len - real, req.rid, kv_committed_len, real,
            )
    elif kv_committed_len > 0:
        _log.warning(
            "ALL-ZERO row on release (rid=%s) commit=%d - %d allocated pages gone "
            "(alloc-vs-write mismatch); pool stays short by that amount",
            req.rid, kv_committed_len, kv_committed_len,
        )

    # Safety net (v1): commit=0/alloc=0 but the row still holds pages.
    kv_allocated_len = req.kv.kv_allocated_len
    if req.kv_committed_len == 0 and kv_allocated_len == 0:
        owned = _owned_pages(self, req)
        if owned > 0:
            row = self.req_to_token_pool.req_to_token[req.req_pool_idx]
            pages = row[:owned]
            pages = pages[pages > 0]
            if pages.numel() > 0:
                self.token_to_kv_pool_allocator.free(pages)
                req._kv_leak_repaired = True
                _log.warning(
                    "REPAIRED %d leaked KV pages (rid=%s) commit=%d alloc=%d",
                    pages.numel(), req.rid,
                    req.kv_committed_len, kv_allocated_len,
                )
        elif owned < 0:
            _log.warning(
                "unsafe-to-repair release (rid=%s) commit=%d alloc=%d "
                "row still owns pages - pages remain leaked; read kv_leak_probe output",
                req.rid, req.kv_committed_len, kv_allocated_len,
            )
    return None


def patch() -> None:
    """Apply the v4 fix. Idempotent. Disabled by SGLANG_KVLEAK_FIX=0.

    The write_cache_indices wrap is applied lazily via a post-import hook
    because importing serving-stack modules here — during sglang_fl import,
    before torch.distributed init — disturbs the txda device context (FlagCX
    picks device=cpu, then the fallback all_reduce crashes: SIGQUIT).  Only
    ChunkCache is imported eagerly — proven safe in v3.
    """
    global _patched
    if _patched:
        return
    if not _enabled():
        _log.skipped("disabled by %s=0", _ENABLED_KEY)
        return

    # ---- 2) Anti-pollution + safety net: rewrite ChunkCache.cache_finished_req release segment ----
    from sglang.srt.mem_cache.chunk_cache import ChunkCache

    ChunkCache.cache_finished_req = _safe_cache_finished_req
    _patched = True
    _log.applied(
        "hardened ChunkCache.cache_finished_req (zero-filter + REPAIRED safety net)"
    )

    # ---- 0) Root cause (v4): force write_cache_indices onto torch fallback (deferred until allocation is imported) ----
    _install_post_import_wrap(_WRITE_INDICES_MODULE, _safe_write_indices_wrap)
