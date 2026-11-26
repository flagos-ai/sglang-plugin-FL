"""TEMP experiment hook (validation only, never merge).

Two SGLang operators are reached on HCU although the platform has no
``sgl_kernel``: ``moe_align_block_size`` (Triton fused_experts, every MoE
model) and ``fast_topk`` (MTP draft selection). The plugin's import guard
answers ``import sgl_kernel`` with a placeholder whose members raise when
called. This hook gives the PLACEHOLDER real implementations:

  fast_topk             -> torch.topk (same values/indices, same shapes for
                           every k, unlike the keepdim max shortcut)
  moe_align_block_size  -> FlagGems' Triton moe_align_block_size
  silu_and_mul          -> SGLang's Triton activation (the kernel the dense
                           MLP already uses on HCU, see activation.py)
  moe_sum_reduce        -> SGLang's Triton moe_sum_reduce_triton
  build_tree_kernel_efficient -> SGLang's Triton tree builder (XPU path)
  verify_tree_greedy    -> SGLang's Triton verify_tree_greedy_triton (XPU path)
  top_k_renorm_prob /   -> SGLang's Triton renormalisation kernels
  top_p_renorm_prob

Round 1 (fast_topk + moe_align_block_size only) got past moe_align and then
hit ``sgl_kernel.silu_and_mul`` inside Triton fused_experts, hence this table.
Not implemented: tree_speculative_sampling_target_only (non-greedy MTP).

A really installed sgl_kernel is never touched: the hook only acts on a
module that carries the plugin's ``__sglang_fl_placeholder__`` marker.
Installed into the job container by setup.sh through a .pth line, so every
Python process of the job (scheduler and TP workers included) gets it.
"""

from __future__ import annotations

import builtins
import sys

TAG = "[hcu-fallback-probe]"


def fast_topk(values, topk, dim):
    import torch

    return torch.topk(values, topk, dim=dim)


def moe_align_block_size(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    experts_ids,
    num_tokens_post_pad,
    cumsum_buffer,
    pad_sorted_token_ids=False,
    ignore_invalid_expert=False,
):
    """SGLang's low-level signature: fill the caller's buffers, return None.

    No host read of a device tensor, so the function is CUDA-graph safe.
    """
    import flag_gems

    if ignore_invalid_expert:
        raise RuntimeError(
            "FlagGems moe_align_block_size has no ignore_invalid_expert form"
        )
    sorted_ids, expert_ids, num_post_pad = flag_gems.moe_align_block_size(
        topk_ids, block_size, num_experts
    )
    pad_value = topk_ids.numel()
    # SGLang sizes its buffers its own way (tiny inputs get numel * block);
    # copy the overlapping prefix and keep the rest as padding.
    sorted_token_ids.fill_(pad_value)
    tokens = min(sorted_token_ids.numel(), sorted_ids.numel())
    sorted_token_ids[:tokens].copy_(sorted_ids[:tokens])
    blocks = min(experts_ids.numel(), expert_ids.numel())
    experts_ids[:blocks].copy_(expert_ids.reshape(-1)[:blocks])
    num_tokens_post_pad.copy_(num_post_pad.reshape(num_tokens_post_pad.shape))


def _act_and_mul(name: str, input, out=None):
    import sglang.kernels.ops.activation.activation as activation

    return getattr(activation, name)(input, out)


def silu_and_mul(input, out=None):
    return _act_and_mul("silu_and_mul", input, out)


def gelu_and_mul(input, out=None):
    return _act_and_mul("gelu_and_mul", input, out)


def gelu_tanh_and_mul(input, out=None):
    return _act_and_mul("gelu_tanh_and_mul", input, out)


def moe_sum_reduce(input, output, routed_scaling_factor):
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import moe_sum_reduce_triton

    return moe_sum_reduce_triton(input, output, routed_scaling_factor)


def top_k_renorm_prob(probs, top_k):
    from sglang.kernels.ops.sampling.renorm_triton import top_k_renorm_probs_triton

    return top_k_renorm_probs_triton(probs, top_k)


def top_p_renorm_prob(probs, top_p):
    from sglang.kernels.ops.sampling.renorm_triton import top_p_renorm_probs_triton

    return top_p_renorm_probs_triton(probs, top_p)


def build_tree_kernel_efficient(*args, **kwargs):
    """Run SGLang's own wrapper, but with the Triton builder underneath
    (the builder SGLang itself selects on XPU)."""
    from sglang.srt.speculative import eagle_utils

    original = eagle_utils.sgl_build_tree_kernel_efficient
    eagle_utils.sgl_build_tree_kernel_efficient = (
        eagle_utils.sgl_build_tree_kernel_triton
    )
    try:
        return eagle_utils.build_tree_kernel_efficient(*args, **kwargs)
    finally:
        eagle_utils.sgl_build_tree_kernel_efficient = original


def verify_tree_greedy(predicts, accept_index, accept_token_num, candidates, **tree):
    from sglang.srt.speculative.eagle_utils import verify_tree_greedy_triton

    # SGLang passes the four tree tensors as keywords, in this order.
    return verify_tree_greedy_triton(
        predicts, accept_index, accept_token_num, candidates, *tree.values()
    )


# sgl_kernel name -> implementation. Names absent here keep raising.
FALLBACKS = {
    "fast_topk": fast_topk,
    "moe_align_block_size": moe_align_block_size,
    "silu_and_mul": silu_and_mul,
    "gelu_and_mul": gelu_and_mul,
    "gelu_tanh_and_mul": gelu_tanh_and_mul,
    "moe_sum_reduce": moe_sum_reduce,
    "top_k_renorm_prob": top_k_renorm_prob,
    "top_p_renorm_prob": top_p_renorm_prob,
    "build_tree_kernel_efficient": build_tree_kernel_efficient,
    "verify_tree_greedy": verify_tree_greedy,
}


def patch_placeholder(name: str) -> bool:
    module = sys.modules.get(name)
    if module is None or not getattr(module, "__sglang_fl_placeholder__", False):
        return False
    for attr, fn in FALLBACKS.items():
        setattr(module, attr, fn)
    sys.stderr.write(
        f"{TAG} fallbacks installed on the sgl_kernel placeholder: "
        f"{', '.join(sorted(FALLBACKS))}\n"
    )
    return True


def install() -> None:
    original_import = builtins.__import__
    patched: set[str] = set()

    def importing(name, globals=None, locals=None, fromlist=(), level=0):
        result = original_import(name, globals, locals, fromlist, level)
        root = "sgl_kernel"
        if root not in patched and root in sys.modules:
            try:
                if patch_placeholder(root):
                    patched.add(root)
            except Exception as error:  # noqa: BLE001 - a probe must not kill its host
                sys.stderr.write(f"{TAG} could not install fallbacks: {error!r}\n")
                patched.add(root)
        return result

    builtins.__import__ = importing
