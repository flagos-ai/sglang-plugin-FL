# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SGLang 0.5.18 import declarations; all kernel calls fail explicitly."""

from _iluvatar_empty_compat import bind_unavailable as _bind_unavailable

__all__ = []

_bind_unavailable(
    globals(),
    (
        "min_p_sampling_from_probs",
        "top_k_top_p_sampling_from_probs",
    ),
)
del _bind_unavailable
