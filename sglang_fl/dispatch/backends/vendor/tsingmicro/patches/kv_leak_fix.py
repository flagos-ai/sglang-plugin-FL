# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardened KV-page release for the sglang token_to_kv_pool leak. (v3 → v4)

补丁解决的问题 (problem this patch solves)
========================================
30000 服务 KV 页池 (token_to_kv_pool) 出现 KV 页"永久扣住"的记账层泄露。
失败/异常请求释放后 available 纹丝不动，池子缺口随时间增长（-374 → -9617）。

v4 变更（2026-09-24，真正的根因修复）
====================================
v3 的两个机制（free_group 兜底结算 + 释放段 0 条目过滤）都只在"释放侧"打补丁，
没有阻止泄露页的产生，所以补丁"没生效"——缺口仍从 -374 涨到 -9617。

v4 找到并修掉了"写入侧"的根因：

根因（已用代码 + 探针日志锁定）
----------------------------
``mem_cache/common.py::write_cache_indices`` 里，当 attention backend 走 Triton
时（``support_triton(backend)`` 为真），prefill 的 extend/prefix 索引用 Triton
内核 ``write_req_to_token_pool_triton`` 写入 req_to_token 行。该内核的
**prefix 写入段**：

    prefix_tensor = tl.load(prefix_tensors + pid).to(tl.pointer_type(tl.int64))

是把一个 uint64 的 data_ptr 当值加载后再强转成 int64 指针，在 txda 上这段
指针强转**静默失效**：多 chunk 请求从第 2 个 chunk 起，prefix 写会把前一
chunk 已写好的行**覆盖成 0**。

旁证（与探针日志完全吻合）：
- decode 的索引是 ``alloc_for_decode`` 里 ``req_to_token_pool.write(...)``
  直接 torch 写（不走 Triton 内核）→ 所以 output 页永远在行里（日志里
  row_nnz 始终 ≥ out）。
- 单 chunk 请求 prefix_len=0，不触发 prefix 写 → 行完整（单 chunk 正常）。
- 多 chunk 请求从 chunk2 起触发 prefix 写 → 前面 chunk 的索引被清零 →
  释放时 ``row[:commit]`` 里只剩"最后一块 + output"，前面 512 页(=2×256
  chunk)变成孤儿页，available 永久缺口。日志 ``origin=530 → row_nnz=50
  =18(末块)+32(output)`` 正是这个现场。

修复（v4）
---------
把 ``write_cache_indices`` 强制走 **非 Triton 的 torch 回退循环**（直接
``req_to_token_pool.write`` 写行，等价于非 triton backend 的既有正确路径），
彻底绕开 txda 上失效的 Triton prefix 指针强转。这样 prefix/extend 索引逐
chunk 都被正确持久化，孤儿页不再产生。

代价：prefill 多几次 ``.item()`` CPU 同步（非 triton backend 本就如此），
功能正确优先，对 prefill 吞吐影响可忽略。

其余（保留 v2/v3 的既有加固）
----------------------------
1) cache_finished_req 释放段 0 条目过滤 —— 防"记账分配未写入"的假页进 free 表。
2) commit=0/alloc=0 但 row 仍挂页时的 REPAIRED 兜底。

（v3 的 Scheduler try/finally free_group_end 兜底已在 v4 移除：free_group_end()
非幂等 —— 它只置 is_not_in_free_group=True，不清理 self.free_group，重复调用会
把同一批页 free 两次，直接造成 pool 记账 available > total 的双释放崩溃，反而被
严格内存检查判成"泄漏"而 raise ValueError。释放侧兜底已由写入侧治本取代。）

怎么开 (enable)
===============
默认不生效。置 SGLANG_KVLEAK_FIX=1 重启服务后才开启。
建议同时开 SGLANG_KVLEAK_PROBE=1：patches/__init__.py 里 fix 先 apply、probe 后
apply（探针 wrap 在 fix 之上，日志完整）。

导入顺序教训（v3 教训的再次强化，v4 已据此改为全延迟）
---------------------------------------------------
v3 崩溃根因：在 patch()（sglang_fl 导入期、torch.distributed 初始化之前）直接
import `sglang.srt.managers.scheduler_output_processor_mixin`，链式拉进
schedule_batch / layers.* / routed_experts_capturer 整套 serving 栈，扰乱 txda
设备上下文 -> FlagCX 拿 cpu -> 回退 torch.distributed -> 首个 all_reduce 抛
`No backend type associated with device type txda` -> SIGQUIT。

v4 早期版本在 patch() 里直接 import `sglang.srt.mem_cache.common`，其链
`memory_pool -> layers.radix_attention -> model_executor.breakable_cuda_graph.*`
同样是导入期拉 serving 栈的危险动作（虽不一定精确命中 routed_experts_capturer，
但属同一类风险）。**v4 现已把 common 改为 sys.meta_path 后置导入钩子延迟应用**：
等 sglang 自己 import 该模块（dist 初始化之后、任何 prefill 之前）时再替换
`write_cache_indices`。导入期不再 import 任何 serving 栈模块，启动安全。
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
# write_cache_indices 定义在 sglang.srt.mem_cache.common；延迟到 sglang 自己
# import 该模块后再替换，避免导入期拉 serving 栈（v3 教训）。
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
        return 0  # 确实没分配页 —— 没什么可补，正常空操作

    bound = len(req.origin_input_ids) + len(req.output_ids)
    if bound <= 0:
        _log.warning("skip repair rid=%s abnormal bound=%d (no prompt?)", req.rid, bound)
        return -1

    if (row[bound:] > 0).any():
        # 槽位被复用且 row 尾部是上一个请求的陈旧页（已在 free 表里），
        # 重新释放会双释放 -> 宁可保留泄露也不冒险。
        _log.warning(
            "skip repair rid=%s unsafe stale-tail beyond bound=%d (row_len=%d)",
            req.rid, bound, row.shape[0],
        )
        return -1

    return int(nnz[:bound].sum().item())


# ─── 0) 治本（v4）：write_cache_indices 强制走 torch 回退，绕开 txda 失效的 Triton ──


def _apply_write_indices_wrap() -> None:
    """Replace common.write_cache_indices with the non-Triton torch fallback.

    The Triton kernel's prefix-write segment (uint64 data_ptr -> int64 pointer
    cast) silently fails on txda and zeroes earlier prefill chunks' indices on
    multi-chunk requests. The torch fallback loop is the same code path already
    proven correct on non-triton backends. Idempotent.

    Applied lazily via post-import hook (see _install_post_import_wrap) — NOT
    called at patch() time, to avoid importing the serving stack too early.
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


# ─── 2) 防污染 + 兜底：重写 ChunkCache.cache_finished_req 释放段 ───────────────


def _safe_cache_finished_req(self, req, is_insert=True):
    kv_committed_len = req.pop_committed_kv_cache()
    row = self.req_to_token_pool.req_to_token[req.req_pool_idx]
    kv_indices = row[:kv_committed_len]
    # 只释放真实页（>0）。row 里 0 = "记账分配但未写入"的洞：不释放，
    # 也绝不放进 free 表（否则 index 0 重复 -> 未来 alloc 双分配数据错乱）。
    # 注：页索引从 1 起（allocator 里 free_pages = arange(1, size+1)），
    # 0 是保留槽，故 >0 判定即"真页"，无误伤。
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

    # 兜底（v1）：commit=0/alloc=0 但 row 仍挂页（reset_for_retract 清零路径）
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

    Import-order note (2026-09-24, v3 crash + v4 fix):
    importing serving-stack modules at patch() time — during sglang_fl import,
    BEFORE torch.distributed init — disturbs the txda device context so FlagCX
    initializes with device=cpu and the fallback all_reduce crashes at startup
    (SIGQUIT). So the write_cache_indices wrap is applied lazily via a
    post-import hook: it fires the moment sglang itself imports the module
    during normal startup (after dist init, before any prefill). Only ChunkCache
    is imported eagerly here — proven safe in v3.
    """
    global _patched
    if _patched:
        return
    if not _enabled():
        _log.skipped("disabled (set SGLANG_KVLEAK_FIX=1 to enable)")
        return

    # ---- 2) 防污染 + 兜底：重写 ChunkCache.cache_finished_req 释放段 ----
    from sglang.srt.mem_cache.chunk_cache import ChunkCache

    ChunkCache.cache_finished_req = _safe_cache_finished_req
    _patched = True
    _log.applied(
        "hardened ChunkCache.cache_finished_req (zero-filter + REPAIRED safety net)"
    )

    # ---- 0) 治本（v4）：write_cache_indices 强制 torch 回退（延迟到 common 被导入时）----
    _install_post_import_wrap(_COMMON_MODULE, _apply_write_indices_wrap)
