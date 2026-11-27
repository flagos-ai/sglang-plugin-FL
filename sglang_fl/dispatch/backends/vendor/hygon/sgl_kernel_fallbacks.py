# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Operators SGLang imports from ``sgl_kernel`` that HCU reaches at runtime.

The Hygon Runtime ships no ``sgl_kernel`` wheel. The plugin's import guard
lets ``import sgl_kernel`` succeed with a placeholder, which is enough for the
operators HCU dispatches to its own implementations. A few call sites are not
routed through the dispatch table and reach the placeholder directly:

* Triton fused-experts (every MoE model): ``moe_align_block_size``,
  ``silu_and_mul`` and the other gated activations;
* EAGLE/MTP drafting and verification: ``fast_topk``,
  ``build_tree_kernel_efficient`` and ``verify_tree_greedy``.

Each gets an implementation here that is already on the platform: plain
PyTorch, FlagGems, or the Triton kernel SGLang itself selects on its non-CUDA
backends. Nothing downloads, compiles or replaces a Runtime component. A
really installed ``sgl_kernel`` always wins: the import guard stays out of its
way, and so does this table.

Not implemented: ``tree_speculative_sampling_target_only``. MTP with sampling
(temperature above zero) still raises the placeholder error on HCU; greedy
verification is covered.

Every function returns plain tensors or fills the caller's buffers and never
reads a device tensor on the host, so the CUDA-graph safe property of the
originals is kept.
"""

from __future__ import annotations

from importlib import import_module


def fast_topk(values, topk, dim):
    """Same values and indices as ``torch.topk``, for every ``topk``."""
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

    ``cumsum_buffer`` is an output of the original that SGLang allocates and
    never reads, so it is left alone. ``pad_sorted_token_ids`` is accepted for
    signature compatibility; the FlagGems output is used as returned.
    """
    import flag_gems

    if ignore_invalid_expert:
        raise RuntimeError(
            "FlagGems moe_align_block_size has no ignore_invalid_expert form"
        )
    sorted_ids, expert_ids, num_post_pad = flag_gems.moe_align_block_size(
        topk_ids, block_size, num_experts
    )
    # SGLang sizes its buffers its own way, so the two shapes only overlap in
    # a prefix: copy that prefix and keep the rest as padding. The padding
    # value is the one SGLang's kernels skip.
    pad_value = topk_ids.numel()
    sorted_token_ids.fill_(pad_value)
    tokens = min(sorted_token_ids.numel(), sorted_ids.numel())
    sorted_token_ids[:tokens].copy_(sorted_ids[:tokens])
    blocks = min(experts_ids.numel(), expert_ids.numel())
    experts_ids[:blocks].copy_(expert_ids.reshape(-1)[:blocks])
    num_tokens_post_pad.copy_(num_post_pad.reshape(num_tokens_post_pad.shape))


def _act_and_mul(name: str, input, out=None):
    activation = import_module("sglang.kernels.ops.activation.activation")

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


def build_tree_kernel_efficient(
    parent_list,
    selected_index,
    verified_seq_len,
    tree_mask,
    positions,
    retrieve_index,
    retrieve_next_token,
    retrieve_next_sibling,
    topk,
    depth,
    draft_token_num,
    tree_mask_mode,
):
    """Fill buffers SGLang allocated: launch SGLang's Triton tree builder."""
    from sglang.srt.speculative.eagle_utils import sgl_build_tree_kernel_triton

    return sgl_build_tree_kernel_triton(
        parent_list,
        selected_index,
        verified_seq_len,
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        topk,
        depth,
        draft_token_num,
        tree_mask_mode,
    )


# SGLang names these keyword arguments at its call site (upstream spelling).
_TREE_KEYWORDS = (
    "retrive_index",
    "retrive_next_token",
    "retrive_next_sibling",
    "target_predict",
)


def verify_tree_greedy(predicts, accept_index, accept_token_num, candidates, **tree):
    """Greedy EAGLE verification with SGLang's Triton kernel, in place."""
    from sglang.srt.speculative.eagle_utils import verify_tree_greedy_triton

    if set(tree) != set(_TREE_KEYWORDS):
        raise TypeError(
            "verify_tree_greedy() expects the keyword arguments "
            f"{', '.join(_TREE_KEYWORDS)}, got {sorted(tree)}"
        )
    return verify_tree_greedy_triton(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        *(tree[key] for key in _TREE_KEYWORDS),
    )


# ``sgl_kernel`` name -> implementation. Names absent here keep raising the
# placeholder error.
SGL_KERNEL_FALLBACKS = {
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
