# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MUSA communicator fusion under the unified operator policy."""


def allreduce_rms_norm_musa(group, norm_module, x, residual, weight):
    fused_result = group.fused_allreduce_rmsnorm(
        x, residual, weight, norm_module.variance_epsilon
    )
    if fused_result is not None:
        return fused_result

    # Unsupported shapes, missing JIT dependencies, or an explicitly disabled
    # communicator retain the pre-patch all-reduce then RMSNorm semantics.
    x = group.all_reduce(x)
    return norm_module.forward(x, residual, None)
