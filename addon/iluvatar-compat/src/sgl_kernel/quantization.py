# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SGLang 0.5.18 import declarations; all kernel calls fail explicitly."""

from _iluvatar_empty_compat import bind_unavailable as _bind_unavailable

__all__ = []

_bind_unavailable(
    globals(),
    (
        "ggml_dequantize",
        "ggml_moe_a8",
        "ggml_moe_a8_vec",
        "ggml_moe_get_block_size",
        "ggml_mul_mat_a8",
        "ggml_mul_mat_vec_a8",
    ),
)
del _bind_unavailable
