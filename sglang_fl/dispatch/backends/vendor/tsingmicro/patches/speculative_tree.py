# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Use SGLang 0.5.18's own Triton EAGLE tree kernels on TsingMicro TXDA.

``sglang.srt.speculative.eagle_utils`` dispatches two speculative-decoding
helpers by device, and TXDA is in none of the branches:

* ``build_tree_kernel_efficient`` -> ``sgl_build_tree_kernel_efficient``, a
  CUDA ``sgl_kernel`` symbol that is only imported on CUDA/HIP/MUSA. On TXDA the
  name is undefined and the first MTP request fails with ``NameError``.
* ``verify_tree_greedy_func`` has CUDA/CPU/NPU/XPU branches only; on TXDA it
  silently does nothing, leaving the accept counters unset.

SGLang already ships device-independent Triton versions for its XPU route
(``sgl_build_tree_kernel_triton``, ``verify_tree_greedy_triton``). Bind those for
TXDA, keeping the upstream buffers and return contract -- the same approach as
``vendor/iluvatar/patches/speculative_tree.py``. Both helpers are only called
from inside ``eagle_utils``, so rebinding its module globals is sufficient.

Applied lazily, right after SGLang itself imports ``eagle_utils`` (no early
import of the serving stack). The signatures are checked against SGLang
0.5.18; on a mismatch the patch is not applied and the failure is logged.
"""

import functools
import inspect

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger
from sglang_fl.dispatch.backends.vendor.tsingmicro.patches.kv_leak_fix import (
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


def _apply() -> None:
    try:
        import sglang.srt.speculative.eagle_utils as m

        build_triton = getattr(m, "sgl_build_tree_kernel_triton", None)
        verify_triton = getattr(m, "verify_tree_greedy_triton", None)
        original_verify = getattr(m, "verify_tree_greedy_func", None)
        if not _signature_is(build_triton, _TREE_PARAMETERS):
            _log.failed("sgl_build_tree_kernel_triton signature differs from SGLang 0.5.18; not patched")
            return
        if not _signature_is(verify_triton, _VERIFY_PARAMETERS):
            _log.failed("verify_tree_greedy_triton signature differs from SGLang 0.5.18; not patched")
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
            verify_triton(
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
        _log.applied("EAGLE tree build / greedy verify -> SGLang Triton kernels (XPU route)")
    except Exception as exc:  # noqa: BLE001
        _log.failed("cannot patch %s: %s", _MODULE, exc)


def patch() -> None:
    """Install the lazy patch. Idempotent. txda-only (called from the txda block)."""
    global _patched
    if _patched:
        return
    _install_post_import_wrap(_MODULE, _apply)
    _patched = True
