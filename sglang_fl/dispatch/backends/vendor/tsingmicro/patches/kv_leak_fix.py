# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardened KV-page release for the sglang ``token_to_kv_pool`` leak (v3 -> v4).

On txda, multi-chunk prefill requests orphan KV pages and permanently drain the
pool.  Off by default — set ``SGLANG_KVLEAK_FIX=1`` (optionally
``SGLANG_KVLEAK_PROBE=1`` for diagnostics) to enable.  Each helper below
documents the part of the fix it implements.
"""

import importlib.abc
import importlib.util
import logging
import os
import sys

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger

logger = logging.getLogger(__name__)
_log = patch_logger("kv_leak_fix")

_patched = False
_ENABLED_KEY = "SGLANG_KVLEAK_FIX"
# write_cache_indices lives in sglang.srt.mem_cache.common; replace it lazily
# once sglang itself imports that module, to avoid pulling the serving stack
# at import time (the v3 lesson).
_COMMON_MODULE = "sglang.srt.mem_cache.common"


def _enabled() -> bool:
    v = os.getenv(_ENABLED_KEY, "")
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
    """Replace common.write_cache_indices with the non-Triton torch fallback.

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
    from sglang.srt.mem_cache import common

    if getattr(common.write_cache_indices, "_kvleak_fallback", False):
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
        # Byte-for-byte replica of the non-triton fallback in common.py.
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
    common.write_cache_indices = _fallback_write_cache_indices
    _log.applied("forced non-Triton write_cache_indices (txda Triton prefix-write bypass)")


class _PostImportLoader(importlib.abc.Loader):
    """Forward to the original loader, then run a callback after the module loads."""

    def __init__(self, inner, callback):
        self._inner = inner
        self._callback = callback

    def create_module(self, spec):
        try:
            return self._inner.create_module(spec)
        except AttributeError:
            return None

    def exec_module(self, module):
        self._inner.exec_module(module)
        self._callback()


class _PostImportWrapFinder(importlib.abc.MetaPathFinder):
    """One-shot finder: wrap a target module's loader to run `callback` after import."""

    def __init__(self, target, callback):
        self._target = target
        self._callback = callback
        self._finding = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self._target or self._finding:
            return None
        self._finding = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._finding = False
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PostImportLoader(spec.loader, self._callback)
        return spec


def _install_post_import_wrap(module_name, callback) -> None:
    """Apply `callback` once `module_name` has been imported by sglang itself.

    If the module is already loaded, apply immediately. Otherwise install a
    meta_path finder that fires the callback right after the module loads.
    """
    if module_name in sys.modules:
        callback()
        return
    for f in sys.meta_path:
        if type(f) is _PostImportWrapFinder and f._target == module_name:
            return
    sys.meta_path.insert(0, _PostImportWrapFinder(module_name, callback))


# ─── 2) Anti-pollution + safety net: rewrite the ChunkCache.cache_finished_req release segment ───


def _safe_cache_finished_req(self, req, is_insert=True):
    """Release-side hardening (retained from v2/v3).

    (1) Free only real pages (index > 0): a 0 is an "accounted-as-allocated
    but never written" hole that must not enter the free table — 0 is a
    reserved slot, and freeing it would let a future alloc double-assign it.
    (2) REPAIRED safety net: when commit=0/alloc=0 but the row still holds
    pages (the reset_for_retract zeroing path), free them so the leak closes
    instead of growing.
    """
    kv_committed_len = req.pop_committed_kv_cache()
    row = self.req_to_token_pool.req_to_token[req.req_pool_idx]
    kv_indices = row[:kv_committed_len]
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
    if req.kv_committed_len == 0 and req.kv_allocated_len == 0:
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
                    req.kv_committed_len, req.kv_allocated_len,
                )
        elif owned < 0:
            _log.warning(
                "unsafe-to-repair release (rid=%s) commit=%d alloc=%d "
                "row still owns pages - pages remain leaked; read kv_leak_probe output",
                req.rid, req.kv_committed_len, req.kv_allocated_len,
            )
    return None


def patch() -> None:
    """Apply the v4 fix. Idempotent. No-op unless SGLANG_KVLEAK_FIX=1.

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
        _log.skipped("disabled (set SGLANG_KVLEAK_FIX=1 to enable)")
        return

    # ---- 2) Anti-pollution + safety net: rewrite ChunkCache.cache_finished_req release segment ----
    from sglang.srt.mem_cache.chunk_cache import ChunkCache

    ChunkCache.cache_finished_req = _safe_cache_finished_req
    _patched = True
    _log.applied(
        "hardened ChunkCache.cache_finished_req (zero-filter + REPAIRED safety net)"
    )

    # ---- 0) Root cause (v4): force write_cache_indices onto torch fallback (deferred until common is imported) ----
    _install_post_import_wrap(_COMMON_MODULE, _apply_write_indices_wrap)
