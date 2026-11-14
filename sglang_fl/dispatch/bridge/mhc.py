# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""SGLang mHC wrappers that route execution through OpManager."""

from __future__ import annotations

import torch

from sglang_fl.dispatch import call_op_selected


def mhc_pre_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
    n_splits_pre: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return meta-compatible outputs for graph tracing."""
    del (
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        n_splits_pre,
    )
    num_tokens, hc_mult, hidden_size = residual.shape
    post_mix = residual.new_empty(num_tokens, hc_mult, 1, dtype=torch.float32)
    comb_mix = residual.new_empty(
        num_tokens, hc_mult, hc_mult, dtype=torch.float32
    )
    layer_input = residual.new_empty(
        num_tokens, hidden_size, dtype=torch.bfloat16
    )
    return post_mix, comb_mix, layer_input


def mhc_pre_bridge(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
    n_splits_pre: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return call_op_selected(
        "mhc_pre",
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        n_splits_pre,
    )


def mhc_post_bridge(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return call_op_selected("mhc_post", x, residual, post_layer_mix, comb_res_mix)
