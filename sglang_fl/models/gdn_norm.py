# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""MUSA GDN RMSNorm + output-gate fusion through FlagGems-sglang."""

import torch

from sglang_fl.dispatch import resolve_op


def gated_rmsnorm_oot(instance, x, z=None):
    # Preserve native semantics for grouped norm, pre-norm gating, missing gate,
    # reshaped inputs and dtypes outside the verified BF16 serving workload.
    if (
        not getattr(instance, "_fl_qwen38", False)
        or z is None
        or x.ndim != 2
        or z.shape != x.shape
        or x.shape[-1] == 0
        or x.device.type != "musa"
        or z.device != x.device
        or x.dtype != torch.bfloat16
        or z.dtype != x.dtype
        or instance.weight.device != x.device
        or instance.weight.ndim != 1
        or instance.weight.numel() != x.shape[-1]
        or not instance.weight.is_contiguous()
        or instance.weight.dtype != x.dtype
        or instance.bias is not None
        or instance.group_size not in (None, x.shape[-1])
        or not instance.norm_before_gate
        or instance.activation not in ("sigmoid", "swish", "silu")
    ):
        return instance.forward_native(x, z)
    return resolve_op("fla_layernorm_gated")(
        x,
        z,
        instance.weight,
        None,
        activation=instance.activation,
        eps=instance.eps,
        is_rms_norm=True,
    )


def register_hooks():
    from sglang.kernels.fused_op import BaseFusedOp
    from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm

    BaseFusedOp.register_oot_forward(RMSNorm, gated_rmsnorm_oot, "oot")
