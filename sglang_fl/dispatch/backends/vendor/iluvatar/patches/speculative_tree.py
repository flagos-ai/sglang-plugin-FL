# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Use SGLang's official Triton tree kernels on the CoreX CUDA alias."""

import functools
import importlib
import inspect
import logging
import sys


logger = logging.getLogger(__name__)
_applied = False
_MODULE = "sglang.srt.speculative.eagle_utils"
_SPEC_MODULE = "sglang.srt.speculative.spec_utils"
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


def _require_signature(function, parameters):
    try:
        compatible = (
            callable(function)
            and tuple(inspect.signature(function).parameters) == parameters
        )
    except (TypeError, ValueError):
        compatible = False
    if not compatible:
        raise RuntimeError(
            "Iluvatar speculative decoding requires SGLang 0.5.18's "
            f"{parameters} Triton tree contract"
        )


def patch_speculative_tree() -> None:
    """Replace only CoreX's CUDA tree bindings, preserving upstream buffers.

    The empty-runtime sgl_kernel symbols are import declarations, not executable
    CUDA kernels. SGLang 0.5.18 already provides equivalent Triton functions for
    its XPU route. Keep the public tree allocation and greedy return contracts;
    do not replace the common import shim or change other vendors' dispatch.
    """
    global _applied
    from sglang_fl.utils import get_device_info

    info = get_device_info()
    if info is None or info.vendor_name != "iluvatar" or _applied:
        return

    module = importlib.import_module(_MODULE)
    build_triton = getattr(module, "sgl_build_tree_kernel_triton", None)
    verify_triton = getattr(module, "verify_tree_greedy_triton", None)
    original_verify = getattr(module, "verify_tree_greedy_func", None)
    original_build = getattr(module, "sgl_build_tree_kernel_efficient", None)
    spec_utils = importlib.import_module(_SPEC_MODULE)
    common = importlib.import_module("sglang.srt.utils.common")
    fast_topk = getattr(common, "fast_topk", None)
    original_topk = getattr(spec_utils, "fast_topk", None)
    _require_signature(build_triton, _TREE_PARAMETERS)
    _require_signature(verify_triton, _VERIFY_PARAMETERS)
    _require_signature(original_verify, (*_VERIFY_PARAMETERS, "topk"))
    _require_signature(fast_topk, ("values", "topk", "dim"))
    if not callable(original_build):
        raise RuntimeError("Iluvatar requires SGLang's CUDA-alias tree binding")
    if not callable(original_topk):
        raise RuntimeError("Iluvatar requires SGLang's speculative topk binding")

    @functools.wraps(original_verify)
    def verify_tree_greedy_iluvatar(
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
        # Upstream CUDA and XPU both leave the public topk hint unused. The
        # 0.5.18 Triton fallback accepts exactly these eight tensor arguments.
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

    # This identical positional ABI retains FULL_MASK/QLEN_ONLY handling and
    # the official explicit rejection of QLEN_ONLY_BITPACKING.
    module.sgl_build_tree_kernel_efficient = build_triton
    module.verify_tree_greedy_func = verify_tree_greedy_iluvatar
    spec_utils.fast_topk = fast_topk
    for name, consumer in tuple(sys.modules.items()):
        if not name.startswith("sglang.") or consumer is None:
            continue
        if getattr(consumer, "sgl_build_tree_kernel_efficient", None) is original_build:
            consumer.sgl_build_tree_kernel_efficient = build_triton
        if getattr(consumer, "verify_tree_greedy_func", None) is original_verify:
            consumer.verify_tree_greedy_func = verify_tree_greedy_iluvatar
        if (
            name.startswith("sglang.srt.speculative.")
            and getattr(consumer, "fast_topk", None) is original_topk
        ):
            consumer.fast_topk = fast_topk
    _applied = True
    logger.info("Iluvatar: speculative tree kernels and topk use upstream fallbacks")
