# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SGLang 0.5.18 import declarations; all kernel calls fail explicitly."""

from _iluvatar_empty_compat import bind_unavailable as _bind_unavailable

__all__ = []

_bind_unavailable(
    globals(),
    (
        "create_greenctx_stream_by_value",
        "get_sm_available",
    ),
)
del _bind_unavailable
