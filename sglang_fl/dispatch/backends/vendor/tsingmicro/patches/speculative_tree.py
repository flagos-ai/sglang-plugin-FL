# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EAGLE/MTP tree build and greedy verify on TsingMicro TXDA (SGLang 0.5.18).

``sglang.srt.speculative.eagle_utils`` dispatches two speculative-decoding
helpers by device, and TXDA is in none of the branches:

* ``build_tree_kernel_efficient`` -> ``sgl_build_tree_kernel_efficient``, a
  CUDA ``sgl_kernel`` symbol only imported on CUDA/HIP/MUSA. On TXDA the name is
  undefined and the first MTP request fails with ``NameError``. Bind SGLang's
  device-independent Triton build (``sgl_build_tree_kernel_triton``, its XPU
  route), as ``vendor/iluvatar/patches/speculative_tree.py`` does.
* ``verify_tree_greedy_func`` has CUDA/CPU/NPU/XPU branches only; on TXDA it
  silently does nothing. SGLang's Triton verify (``verify_tree_greedy_triton``)
  is NOT used: on TXDA it accepts at most one draft token and writes wrong
  ``accept_index`` entries (a one-card check against the reference below:
  chain with every draft correct -> accept 1 instead of 3, accept_index
  ``[0, 1, 0, -1]``), which corrupts the KV/mamba commit, degrades the output
  into garbage and trips SGLang's KV pool leak check. ``_verify_tree_greedy_torch``
  implements the same contract (CUDA ``verify_tree_greedy``) with batched torch
  ops: one step per tree depth, siblings scanned per step.

Applied lazily, right after SGLang itself imports ``eagle_utils``. The
signatures are checked against SGLang 0.5.18; on a mismatch the patch is not
applied and the failure is logged.
"""

import functools
import inspect

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger
from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._utils import (
    _install_post_import_wrap,
)

_log = patch_logger("speculative_tree")

_MODULE = "sglang.srt.speculative.eagle_utils"
_patched = False

_VERIFY_PARAMETERS = (
    "predicts",
    "accept_index",
    "accept_token_num",
    "candidates",
    "retrieve_index",
    "retrieve_next_token",
    "retrieve_next_sibling",
    "target_predict",
)
_TREE_PARAMETERS = (
    "parent_list",
    "selected_index",
    "verified_seq_len",
    "tree_mask",
    "positions",
    "retrieve_index",
    "retrieve_next_token",
    "retrieve_next_sibling",
    "topk",
    "depth",
    "draft_token_num",
    "tree_mask_mode",
)


def _signature_is(function, parameters) -> bool:
    try:
        return callable(function) and tuple(inspect.signature(function).parameters) == parameters
    except (TypeError, ValueError):
        return False


def _verify_tree_greedy_torch(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrieve_index,
    retrieve_next_token,
    retrieve_next_sibling,
    target_predict,
):
    """Greedy tree verify, same contract as SGLang's CUDA ``verify_tree_greedy``.

    Shapes: candidates / retrieve_* ``[B, N]``; target_predict ``[B, N]`` (indexed
    flat by retrieve index); accept_index ``[B, S]``; accept_token_num ``[B]``;
    predicts ``[B * N]``. Outputs are written in place.
    """
    import torch

    B, N = candidates.shape
    S = accept_index.shape[1]
    dev = candidates.device
    rows = torch.arange(B, device=dev)
    flat_target = target_predict.reshape(-1)
    r_index = retrieve_index.to(torch.int64)
    nxt = retrieve_next_token.to(torch.int64)
    sib = retrieve_next_sibling.to(torch.int64)

    last = r_index[:, 0].clone()  # retrieve index of the last accepted node
    accept_index[:, 0] = last.to(accept_index.dtype)
    num = torch.zeros(B, dtype=torch.int64, device=dev)
    active = torch.ones(B, dtype=torch.bool, device=dev)
    cur = torch.zeros(B, dtype=torch.int64, device=dev)  # node position in the tree row
    for j in range(1, S):
        cur = torch.where(active, nxt[rows, cur.clamp(min=0)], cur)
        target = flat_target[last]
        matched = torch.zeros(B, dtype=torch.bool, device=dev)
        node = cur.clone()
        hit = torch.full((B,), -1, dtype=torch.int64, device=dev)
        for _ in range(N):  # scan siblings: first match wins
            valid = active & ~matched & (node >= 0)
            if not bool(valid.any()):
                break
            safe = node.clamp(min=0)
            ok = valid & (candidates[rows, safe] == target)
            hit = torch.where(ok, safe, hit)
            matched = matched | ok
            node = torch.where(valid & ~ok, sib[rows, safe], node)
        acc = active & matched
        if not bool(acc.any()):
            break
        new_last = r_index[rows, hit.clamp(min=0)]
        predicts[last[acc]] = target[acc].to(predicts.dtype)
        num = num + acc.to(torch.int64)
        accept_index[rows[acc], j] = new_last[acc].to(accept_index.dtype)
        last = torch.where(acc, new_last, last)
        cur = torch.where(acc, hit, cur)
        active = acc
    accept_token_num.copy_(num.to(accept_token_num.dtype))
    predicts[last] = flat_target[last].to(predicts.dtype)


def _apply() -> None:
    try:
        import sglang.srt.speculative.eagle_utils as m

        build_triton = getattr(m, "sgl_build_tree_kernel_triton", None)
        original_verify = getattr(m, "verify_tree_greedy_func", None)
        if not _signature_is(build_triton, _TREE_PARAMETERS):
            _log.failed("sgl_build_tree_kernel_triton signature differs from SGLang 0.5.18; not patched")
            return
        if not _signature_is(original_verify, (*_VERIFY_PARAMETERS, "topk")):
            _log.failed("verify_tree_greedy_func signature differs from SGLang 0.5.18; not patched")
            return

        @functools.wraps(original_verify)
        def verify_tree_greedy_txda(
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
            # topk is an unused hint on every upstream branch.
            _verify_tree_greedy_torch(
                predicts=predicts,
                accept_index=accept_index,
                accept_token_num=accept_token_num,
                candidates=candidates,
                retrieve_index=retrieve_index,
                retrieve_next_token=retrieve_next_token,
                retrieve_next_sibling=retrieve_next_sibling,
                target_predict=target_predict,
            )
            return predicts, accept_index, accept_token_num

        # Same positional ABI as the CUDA binding called at eagle_utils' else-branch.
        m.sgl_build_tree_kernel_efficient = build_triton
        m.verify_tree_greedy_func = verify_tree_greedy_txda
        _log.applied("EAGLE tree build -> SGLang Triton kernel; greedy verify -> torch (txda)")
    except Exception as exc:  # noqa: BLE001
        _log.failed("cannot patch %s: %s", _MODULE, exc)


def patch() -> None:
    """Install the lazy patch. Idempotent. txda-only (called from the txda block)."""
    global _patched
    if _patched:
        return
    _install_post_import_wrap(_MODULE, _apply)
    _patched = True
