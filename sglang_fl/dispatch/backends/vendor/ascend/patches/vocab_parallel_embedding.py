# Copyright 2026 FlagOS Contributors
"""Fuse shard indexing, lookup and zero masking for Ascend TP embedding."""

from __future__ import annotations

import functools
import inspect
import logging
import os

import torch
import triton
import triton.language as tl

try:
    from torch_npu import get_npu_format as _get_npu_format
except ImportError:
    _get_npu_format = None

logger = logging.getLogger(__name__)
_MARKER = "_sglang_fl_fused_vocab_embedding"


@triton.jit
def _vocab_embedding_kernel(
    Ids, Weight, Out,
    H: tl.constexpr,
    ORG_START: tl.constexpr, ORG_END: tl.constexpr,
    ORG_PADDING: tl.constexpr,
    ADDED_START: tl.constexpr, ADDED_END: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    token = tl.load(Ids + row).to(tl.int64)
    org = (token >= ORG_START) & (token < ORG_END)
    added = (token >= ADDED_START) & (token < ADDED_END)
    added_offset = ADDED_START - (ORG_END - ORG_START) - ORG_PADDING
    offset = tl.where(org, ORG_START, 0) + tl.where(added, added_offset, 0)
    index = token - offset
    valid = org | added
    columns = tl.arange(0, BLOCK)
    values = tl.load(
        Weight + index * H + columns,
        mask=valid & (columns < H), other=0,
    )
    tl.store(Out + row * H + columns, values, mask=columns < H)


def fused_vocab_embedding(weight, input_, shard):
    """Inference-only contiguous unquantized embedding, before TP reduction."""
    output = torch.empty(
        (*input_.shape, weight.shape[1]), device=weight.device, dtype=weight.dtype,
    )
    if input_.numel():
        with torch.npu.device(weight.device):
            _vocab_embedding_kernel[(input_.numel(),)](
                input_, weight, output,
                weight.shape[1], shard.org_vocab_start_index,
                shard.org_vocab_end_index, shard.num_org_vocab_padding,
                shard.added_vocab_start_index, shard.added_vocab_end_index,
                triton.next_power_of_2(weight.shape[1]),
            )
    return output


def _supported(layer, input_, unquantized_type):
    weight = getattr(layer, "weight", None)
    return (
        layer.tp_size > 1
        and type(layer.quant_method) is unquantized_type
        and isinstance(weight, torch.Tensor)
        and not torch.is_grad_enabled()
        and input_.device.type == "npu"
        and weight.device == input_.device
        and input_.dtype in (torch.int32, torch.int64)
        and weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and input_.is_contiguous()
        and weight.ndim == 2
        and weight.is_contiguous()
        and _get_npu_format is not None
        and _get_npu_format(weight) == 2
        and _get_npu_format(input_) == 2
        and 0 < weight.shape[1] <= 8192
        and weight.shape[0] >= layer.shard_indices.num_elements_padded
    )


def patch_vocab_parallel_embedding():
    if os.environ.get("SGLANG_FL_ASCEND_FUSED_VOCAB_EMBEDDING", "1") != "1":
        return False
    import sglang.srt.layers.vocab_parallel_embedding as module

    required = (
        "VocabParallelEmbedding", "UnquantizedEmbeddingMethod",
        "use_symmetric_memory", "get_tp_group", "is_allocation_symmetric",
        "get_attn_tp_context", "attn_tp_all_reduce",
        "tensor_model_parallel_all_reduce",
    )
    if not all(hasattr(module, name) for name in required):
        logger.warning("Skipping fused vocabulary embedding: unsupported SGLang API")
        return False
    original = module.VocabParallelEmbedding.forward
    if getattr(original, _MARKER, False):
        return False
    try:
        parameters = tuple(inspect.signature(original).parameters.values())
    except (TypeError, ValueError):
        return False
    if len(parameters) != 2 or any(
        parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ) for parameter in parameters
    ):
        logger.warning("Skipping fused vocabulary embedding: unsupported forward signature")
        return False

    @functools.wraps(original)
    def forward(self, input_):
        if not _supported(self, input_, module.UnquantizedEmbeddingMethod):
            return original(self, input_)
        with module.use_symmetric_memory(
            module.get_tp_group(), disabled=not module.is_allocation_symmetric()
        ):
            output = fused_vocab_embedding(self.weight, input_, self.shard_indices)
        if not module.get_attn_tp_context().input_scattered:
            if self.use_attn_tp_group:
                output = module.attn_tp_all_reduce(output)
            else:
                output = module.tensor_model_parallel_all_reduce(output)
        return output

    setattr(forward, _MARKER, True)
    module.VocabParallelEmbedding.forward = forward
    logger.info("Installed Ascend fused sharded vocabulary embedding")
    return True
