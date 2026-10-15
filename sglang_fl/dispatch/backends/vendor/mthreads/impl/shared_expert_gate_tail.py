# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MUSA gate-tail arithmetic; model and batch guards live in the bridge."""

import torch


def shared_expert_gate_tail_musa(block, hidden):
    from ..moe import shared_expert_gate_tail as kernel

    shared = block.shared_expert(hidden)
    gate = block.shared_expert_gate
    if (
        getattr(gate, "bias", None) is None
        and kernel.triton is not None
        and kernel.candidate_guard_reason(
            hidden,
            gate.weight,
            shared,
            enabled=True,
            capturing=False,
        )
        == "eligible"
    ):
        # Fixed-shape allocation is graph-owned, as in the measured core path;
        # the kernel receives an explicit output and never allocates implicitly.
        output = torch.empty_like(shared)
        return kernel.fused_shared_expert_gate_tail(
            hidden,
            gate.weight,
            shared,
            out=output,
            enabled=True,
        )
    # Reuse the computed MLP output on a guard miss.
    return torch.sigmoid(gate(hidden)) * shared
