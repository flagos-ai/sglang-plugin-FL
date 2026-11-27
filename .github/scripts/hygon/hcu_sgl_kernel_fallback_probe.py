"""TEMP experiment hook (validation only, never merge).

Two SGLang operators are reached on HCU although the platform has no
``sgl_kernel``: ``moe_align_block_size`` (Triton fused_experts, every MoE
model) and ``fast_topk`` (MTP draft selection). The plugin's import guard
answers ``import sgl_kernel`` with a placeholder whose members raise when
called. This hook gives the PLACEHOLDER real implementations:

  fast_topk             -> torch.topk (same values/indices, same shapes for
                           every k, unlike the keepdim max shortcut)
  moe_align_block_size  -> FlagGems' Triton moe_align_block_size

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


def patch_placeholder(name: str) -> bool:
    module = sys.modules.get(name)
    if module is None or not getattr(module, "__sglang_fl_placeholder__", False):
        return False
    module.fast_topk = fast_topk
    module.moe_align_block_size = moe_align_block_size
    sys.stderr.write(f"{TAG} fallbacks installed on the sgl_kernel placeholder\n")
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
