# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Portable PyTorch reference implementations for mHC inference."""

from __future__ import annotations

from typing import Optional

import torch


def _sinkhorn_normalize(value: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    value = value.softmax(dim=-1) + eps
    value = value / (value.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        value = value / (value.sum(dim=-1, keepdim=True) + eps)
        value = value / (value.sum(dim=-2, keepdim=True) + eps)
    return value


def mhc_pre_torch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    n_splits_pre: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute mHC pre-mixing with PyTorch operations."""
    del n_splits, n_splits_pre
    hc_mult = residual.shape[-2]
    outer_shape = residual.shape[:-2]
    residual_flat = residual.reshape(-1, hc_mult, residual.shape[-1])
    residual_projection = residual_flat.flatten(-2, -1).float()
    inv_rms = (
        residual_projection.square().mean(dim=-1, keepdim=True) + rms_eps
    ).rsqrt()
    mixes = (residual_projection @ fn.float().T) * inv_rms
    scale = torch.cat(
        (
            hc_scale[0].expand(hc_mult),
            hc_scale[1].expand(hc_mult),
            hc_scale[2].expand(hc_mult * hc_mult),
        )
    )
    mixes = mixes * scale + hc_base
    pre_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(-1) + hc_pre_eps
    post_mix = (
        mixes[:, hc_mult : 2 * hc_mult].sigmoid() * hc_post_mult_value
    ).unsqueeze(-1)
    comb_res_mix = _sinkhorn_normalize(
        mixes[:, 2 * hc_mult :].reshape(-1, hc_mult, hc_mult),
        sinkhorn_repeat,
        hc_sinkhorn_eps,
    )
    layer_input = (residual_flat.float() * pre_mix).sum(dim=-2).to(residual.dtype)
    return (
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_res_mix.reshape(*outer_shape, hc_mult, hc_mult),
        layer_input.reshape(*outer_shape, residual.shape[-1]),
    )


def mhc_post_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """Compute mHC post-mixing with PyTorch operations."""
    output = x.float().unsqueeze(-2) * post_layer_mix.float()
    output = output + torch.matmul(
        comb_res_mix.transpose(-1, -2).float(), residual.float()
    )
    return output.to(residual.dtype)
