# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Native fallbacks for optional vendor fusions."""

import torch


def shared_expert_gate_tail(block, hidden):
    shared = block.shared_expert(hidden)
    return torch.sigmoid(block.shared_expert_gate(hidden)) * shared


def moe_sum_reduce(original, routed, output, routed_scaling_factor, *args, **kwargs):
    return original(routed, output, routed_scaling_factor, *args, **kwargs)


def allreduce_rms_norm(group, norm_module, x, residual, weight):
    return norm_module.forward(group.all_reduce(x), residual, None)
