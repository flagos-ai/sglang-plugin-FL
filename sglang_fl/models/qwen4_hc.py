# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Qwen4 HC wiring for FlagGems-sglang on MUSA.

Inspired by vllm-plugin-FL #455. Numerical kernels live in FlagGems-sglang;
this module owns SGLang's residual tuple and next-normalization contracts.
Backend selection uses the shared operator dispatch policy.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from sglang_fl.dispatch import resolve_op

logger = logging.getLogger(__name__)


def _eligible(*xs):
    return (
        bool(xs)
        and all(
            x.device.type in ("musa", "privateuseone", "cuda")
            and x.dtype in (torch.bfloat16, torch.float16)
            and x.is_contiguous()
            for x in xs
        )
        and len({x.device for x in xs}) == 1
    )


def _norm(original, self, x):
    if not getattr(self, "_fl_qwen38", False) or not _eligible(x, self.weight):
        return original(self, x)
    group = self.group_size or self.weight.numel()
    if x.shape[-1] != self.weight.numel() or group > 8192:
        return original(self, x)
    return resolve_op("qwen4_grouped_gemma_rmsnorm")(
        x, self.weight, self.weight.numel() // group, self.variance_epsilon
    )


def _mix(original, self, hyper_input, normalized_input=None):
    weights = (self.input_mix_weight_down.weight, self.input_mix_weight_up.weight)
    if not getattr(self, "_fl_qwen38", False) or not _eligible(hyper_input, *weights):
        return original(self, hyper_input, normalized_input)
    if not hyper_input.numel():
        return hyper_input.new_empty((*hyper_input.shape[:-1], self.hidden_size)), (
            hyper_input,
            hyper_input,
            None,
        )
    if normalized_input is not None:
        if (
            not self.config.hc_per_branch_norm
            or normalized_input.shape != hyper_input.shape
        ):
            raise ValueError("Invalid normalized HC input")
        normed = normalized_input
    elif self.config.hc_per_branch_norm:
        normed = self.hc_norm(hyper_input)
    else:
        normed = self.hc_norm(
            hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
        ).flatten(-2)
    injection = None
    if hasattr(self, "_fl_down_inject_weight"):
        packed = F.linear(normed, self._fl_down_inject_weight)
        down, injection = packed.split([self.config.hc_lowrank, self.hc_count], dim=-1)
    else:
        down = F.linear(normed, weights[0])
    activated = resolve_op("qwen4_hc_scale_silu")(down, self.hc_count)
    logits = F.linear(activated, weights[1])
    mixed = resolve_op("qwen4_hc_gate_reduce")(logits, normed, self.hc_count)
    return mixed.to(self.params_dtype), (hyper_input, normed, injection)


def _can_combine(self, block_output, residuals):
    return (
        getattr(self, "_fl_qwen38", False)
        and len(residuals) == 3
        and residuals[2] is not None
        and _eligible(
            block_output, residuals[0], residuals[1], self.block_inject_weight.weight
        )
        and residuals[0].shape[-1] == self.hc_count * self.hidden_size
        and block_output.shape[-1] == self.hidden_size
    )


def _combine(original, self, block_output, residuals):
    if not _can_combine(self, block_output, residuals):
        return original(self, block_output, residuals)
    residual, _, logits = residuals
    return resolve_op("qwen4_hc_inject_combine")(
        logits, block_output, residual, self.hc_count
    ).to(self.params_dtype)


def _combine_norm(original, self, block_output, residuals, next_norm):
    if not _can_combine(self, block_output, residuals):
        return original(self, block_output, residuals, next_norm)
    if (
        next_norm is None
        or next_norm.group_size != self.hidden_size
        or not _eligible(next_norm.weight)
        or next_norm.weight.numel() != self.hc_count * self.hidden_size
    ):
        return _combine(None, self, block_output, residuals), None
    residual, _, logits = residuals
    return resolve_op("qwen4_hc_combine_norm")(
        residual,
        block_output,
        logits,
        next_norm.weight,
        next_norm.variance_epsilon,
        self.hc_count,
    )


def _ple_norm(original, self, x):
    if not getattr(self, "_fl_qwen38", False) or not _eligible(x, self.weight):
        return original(self, x)
    group = self.group_size or self.weight.numel()
    return resolve_op("qwen4_grouped_gemma_rmsnorm")(
        x, self.weight, self.weight.numel() // group, self.eps
    )


def prepare_model(result, self, *args, **kwargs):
    """Bind packed checkpoint views before loading weights for this model."""
    from sglang.srt.layers.hyperconnection import GatedResidual, GroupedGemmaRMSNorm
    from sglang.srt.models.qwen4_exp import Qwen4ExpPLEGroupedNorm
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

    from sglang_fl.dispatch.backends.flagos.qwen4 import OP_NAMES

    from .qwen4_communication import prepare

    for name in OP_NAMES:
        resolve_op(name)
    prepare()
    for module in self.modules():
        module._fl_qwen38 = True
        if isinstance(module, QSAIndexer):
            module._forward_method = module.forward_cuda
        if isinstance(module, (GroupedGemmaRMSNorm, Qwen4ExpPLEGroupedNorm)):
            module._jit_group_size = None
        if isinstance(module, GatedResidual):
            module._jit_mix_ok = False
            module._jit_combine_ok = False
            module._split_combine_ok = False
            module._gate_stream = None
            module._mix_flydsl_ok = False
            if hasattr(module, "input_mix_weight_down") and hasattr(
                module, "block_inject_weight"
            ):
                down = module.input_mix_weight_down.weight
                inject = module.block_inject_weight.weight
                packed = torch.cat((down.detach(), inject.detach()), dim=0)
                module.register_buffer(
                    "_fl_down_inject_weight", packed, persistent=False
                )
                down.data = packed[: down.shape[0]]
                inject.data = packed[down.shape[0] :]


def hooks():
    from sglang.srt.plugins.hook_registry import HookType

    prefix = "sglang.srt.layers.hyperconnection."
    ple = "sglang.srt.models.qwen4_exp.Qwen4ExpPLEGroupedNorm.forward"
    return [
        (
            "sglang.srt.models.qwen4_exp.Qwen4ExpForConditionalGeneration.__init__",
            prepare_model,
            HookType.AFTER,
        ),
        (ple, _ple_norm, HookType.AROUND),
        *[
            (prefix + target, hook, HookType.AROUND)
            for target, hook in (
                ("GroupedGemmaRMSNorm.forward", _norm),
                ("GatedResidual.mix", _mix),
                ("GatedResidual.combine", _combine),
                ("GatedResidual.combine_and_normalize", _combine_norm),
            )
        ],
    ]
