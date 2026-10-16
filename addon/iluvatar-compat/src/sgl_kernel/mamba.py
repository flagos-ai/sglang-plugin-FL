# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SGLang 0.5.18 import declarations; all kernel calls fail explicitly."""

from _iluvatar_empty_compat import bind_unavailable as _bind_unavailable

__all__ = []

_bind_unavailable(
    globals(),
    (
        "causal_conv1d_fn_cpu",
        "causal_conv1d_update_cpu",
        "chunk_gated_delta_rule_cpu",
    ),
)
del _bind_unavailable
