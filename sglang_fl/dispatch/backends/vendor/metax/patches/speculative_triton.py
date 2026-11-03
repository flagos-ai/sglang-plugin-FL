"""Use SGLang's Triton EAGLE tree kernels on MetaX (no sgl_kernel).

sglang.srt.speculative.eagle_utils imports build_tree_kernel_efficient
and verify_tree_greedy from sgl_kernel only when is_cuda() (or HIP /
MUSA) is true, and ships Triton versions for XPU. MetaX reports
is_cuda() == False and has no sgl_kernel, so:

- tree building falls into the else branch and calls an undefined
  sgl_build_tree_kernel_efficient (NameError), and
- verify_tree_greedy_func matches no branch and returns its inputs
  unverified, and
- sglang.kernels.ops.speculative.cache_locs.assign_extend_cache_locs_func
  / ..._uniform_func match no branch and return None, so the verify
  batch has no KV write locations.

Bind them to the Triton implementations the CUDA/XPU branches use. Only the greedy verify path
is covered; non-greedy speculative sampling still needs sgl_kernel.
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

_MODULE = "sglang.srt.speculative.eagle_utils"
_patched = False


def patch_speculative_triton() -> None:
    global _patched
    if _patched:
        return
    try:
        module = importlib.import_module(_MODULE)
    except ImportError:
        return
    if not (
        hasattr(module, "sgl_build_tree_kernel_triton")
        and hasattr(module, "verify_tree_greedy_triton")
    ):
        return  # SGLang version without the Triton EAGLE kernels

    patched = []
    if not hasattr(module, "sgl_build_tree_kernel_efficient"):
        module.sgl_build_tree_kernel_efficient = module.sgl_build_tree_kernel_triton
        patched.append("build_tree")

    verify_triton = module.verify_tree_greedy_triton

    def verify_tree_greedy_func_metax(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        target_predict,
        topk: int = -1,
    ):
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

    module.verify_tree_greedy_func = verify_tree_greedy_func_metax
    patched.append("verify_tree_greedy")
    patched += _patch_cache_locs()
    _patched = True
    logger.info("patched EAGLE tree kernels for MetaX (triton): %s", ", ".join(patched))


_CACHE_LOCS_MODULE = "sglang.kernels.ops.speculative.cache_locs"
# Speculative modules that bind assign_extend_cache_locs_func at import time.
_CACHE_LOCS_IMPORTERS = (
    "sglang.srt.speculative.ngram_worker",
    "sglang.srt.speculative.dflash_worker_v2",
    "sglang.srt.speculative.dspark_components.dspark_kv_inject",
    "sglang.srt.speculative.dspark_components.dspark_planner",
)


def _patch_cache_locs() -> list:
    import sys

    import torch

    try:
        module = importlib.import_module(_CACHE_LOCS_MODULE)
    except ImportError:
        return []
    if not (
        hasattr(module, "assign_extend_cache_locs")
        and hasattr(module, "assign_extend_cache_locs_uniform")
    ):
        return []

    extend_kernel = module.assign_extend_cache_locs
    uniform_kernel = module.assign_extend_cache_locs_uniform
    next_power_of_2 = module.next_power_of_2

    def assign_extend_cache_locs_func_metax(
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        batch_size,
        draft_token_num,
        device,
    ):
        out_cache_loc = torch.empty(
            (batch_size * draft_token_num,), dtype=torch.int64, device=device
        )
        extend_kernel[(batch_size,)](
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            req_to_token.shape[1],
            next_power_of_2(batch_size),
        )
        return out_cache_loc

    def assign_extend_cache_locs_uniform_func_metax(
        req_pool_indices,
        req_to_token,
        start_offset,
        batch_size,
        draft_token_num,
        device,
    ):
        out_cache_loc = torch.empty(
            (batch_size * draft_token_num,), dtype=torch.int64, device=device
        )
        uniform_kernel[(batch_size,)](
            req_pool_indices,
            req_to_token,
            start_offset,
            out_cache_loc,
            req_to_token.shape[1],
            draft_token_num,
        )
        return out_cache_loc

    module.assign_extend_cache_locs_func = assign_extend_cache_locs_func_metax
    module.assign_extend_cache_locs_uniform_func = (
        assign_extend_cache_locs_uniform_func_metax
    )
    for name in _CACHE_LOCS_IMPORTERS:
        importer = sys.modules.get(name)
        if importer is not None and hasattr(importer, "assign_extend_cache_locs_func"):
            importer.assign_extend_cache_locs_func = assign_extend_cache_locs_func_metax
    return ["assign_extend_cache_locs"]
