# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Ascend adapter for `chunk_gated_delta_rule`.
#
# This file mirrors the upstream module path `sgl_kernel_npu.fla.chunk` and
# thin-wraps its `chunk_gated_delta_rule_npu`. In srt_empty mode (no
# sgl_kernel_npu installed), the import below resolves through the stub
# finder to a None-attribute module; the resulting TypeError at call time is
# caught by the plugin dispatch fallback and forwarded to flagos / reference.

from typing import Optional

import torch

from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu


def chunk_gated_delta_rule_ascend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    # Bridges sglang mainline signature (which carries initial_state_indices for
    # per-request state-pool addressing) to the upstream kernel, which only
    # accepts a per-batch initial_state tensor.
    if initial_state is not None and initial_state_indices is not None:
        initial_state = initial_state[initial_state_indices]

    o, final_state, h = chunk_gated_delta_rule_npu(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        head_first=head_first,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    return o, final_state, h
