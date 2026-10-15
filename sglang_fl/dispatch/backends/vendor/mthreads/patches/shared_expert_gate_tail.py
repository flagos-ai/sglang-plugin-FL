# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Attach the M4 shared gate tail without replacing Qwen's router/collectives."""

import importlib
import inspect
import logging
from contextvars import ContextVar
from functools import wraps

import torch

from sglang_fl.dispatch import resolve_op

from ..env import enabled as _enabled
from ..env import is_s5000

logger = logging.getLogger(__name__)
_ENV = "SGLANG_MUSA_SHARED_EXPERT_GATE_TAIL_FUSED"
_MARKER = "_sglang_fl_musa_shared_gate_tail"
_MODEL_MARKER = "_sglang_fl_musa_gate_tail_text_model"
_ACTIVE_BLOCK = ContextVar("musa_shared_gate_tail_block", default=None)


def _none_or_empty_items(value):
    return value is None or (
        type(value) in (list, tuple) and all(item is None for item in value)
    )


def _eligible(module, block, hidden, batch, reduce_scatter, allreduce_fusion):
    if batch is None or not _enabled(_ENV, default=True):
        return False
    try:
        return (
            hidden.device.type == "musa"
            and hidden.dtype == torch.bfloat16
            and tuple(hidden.shape) == (4, 2048)
            and is_s5000(hidden.device)
            and getattr(block, _MODEL_MARKER, False)
            and block.tp_size == 2
            and module.get_pp_group().world_size == 1
            and module.get_moe_data_parallel_world_size() == 1
            and module.get_attn_context_model_parallel_world_size() == 1
            and module.get_moe_a2a_backend().is_none()
            and batch.forward_mode.is_decode()
            and _none_or_empty_items(batch.lora_ids)
            and _none_or_empty_items(batch.mm_inputs)
            and batch.mm_input_embeds is None
            and batch.spec_info is None
            and (batch.spec_algorithm is None or batch.spec_algorithm.is_none())
            and getattr(batch, "deferred_future_token_ids_map", None) is None
            and not getattr(batch, "deferred_text_mrope_positions", False)
            and batch.tbo_split_seq_index is None
            and batch.tbo_parent_token_range is None
            and batch.tbo_padded_len is None
            and batch.tbo_children is None
            and block.alt_stream is None
            and not reduce_scatter
            and not allreduce_fusion
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _shared_forward(block, hidden):
    # Resolve through the common policy/cache without retrying a failed launch.
    return resolve_op("shared_expert_gate_tail")(block, hidden)


def _patch_block(module):
    cls = module.Qwen2MoeSparseMoeBlock
    originals = (cls.__init__, cls.forward, cls._forward_shared_experts)
    markers = [getattr(fn, _MARKER, False) for fn in originals]
    if any(markers):
        return all(markers)
    init, forward, shared = originals
    try:
        init_signature = inspect.signature(init)
        if "config" not in init_signature.parameters:
            return False
        if tuple(inspect.signature(forward).parameters) != (
            "self",
            "hidden_states",
            "forward_batch",
            "use_reduce_scatter",
            "should_allreduce_fusion",
        ) or tuple(inspect.signature(shared).parameters) != ("self", "hidden_states"):
            return False
    except (TypeError, ValueError):
        return False

    @wraps(init)
    def init_with_model_marker(self, *args, **kwargs):
        self.__dict__.pop(_MODEL_MARKER, None)
        result = init(self, *args, **kwargs)
        bound = init_signature.bind(self, *args, **kwargs)
        config = bound.arguments["config"]
        setattr(
            self,
            _MODEL_MARKER,
            getattr(config, "model_type", None) == "qwen3_5_moe_text",
        )
        return result

    @wraps(forward)
    def forward_with_context(
        self,
        hidden_states,
        forward_batch=None,
        use_reduce_scatter=False,
        should_allreduce_fusion=False,
    ):
        eligible = _eligible(
            module,
            self,
            hidden_states,
            forward_batch,
            use_reduce_scatter,
            should_allreduce_fusion,
        )
        token = _ACTIVE_BLOCK.set(self if eligible else None)
        try:
            return forward(
                self,
                hidden_states,
                forward_batch,
                use_reduce_scatter,
                should_allreduce_fusion,
            )
        finally:
            _ACTIVE_BLOCK.reset(token)

    @wraps(shared)
    def shared_with_gate_tail(self, hidden_states):
        if (
            _ACTIVE_BLOCK.get() is self
            and self.shared_expert is not None
            and self.shared_expert_gate is not None
        ):
            return _shared_forward(self, hidden_states)
        return shared(self, hidden_states)

    replacements = (init_with_model_marker, forward_with_context, shared_with_gate_tail)
    for fn in replacements:
        setattr(fn, _MARKER, True)
    names = ("__init__", "forward", "_forward_shared_experts")
    try:
        for name, fn in zip(names, replacements):
            setattr(cls, name, fn)
    except Exception:
        for name, original in zip(names, originals):
            setattr(cls, name, original)
        raise
    return True


def apply_musa_shared_expert_gate_tail_patch():
    if not _enabled(_ENV, default=True):
        return False
    try:
        module = importlib.import_module("sglang.srt.models.qwen2_moe")
        applied = _patch_block(module)
    except (ImportError, AttributeError) as exc:
        logger.warning("MUSA shared gate-tail patch unavailable: %s", exc)
        return False
    if applied:
        logger.info("MUSA M4 shared gate-tail plugin hook installed")
    else:
        logger.warning("MUSA shared gate-tail patch skipped: unsupported Qwen API")
    return applied
