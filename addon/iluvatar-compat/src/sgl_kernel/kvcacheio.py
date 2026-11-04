# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SGLang 0.5.18 import declarations; all kernel calls fail explicitly."""

from _iluvatar_empty_compat import bind_unavailable as _bind_unavailable

__all__ = []

_bind_unavailable(
    globals(),
    (
        "transfer_embedding_ranges_direct",
        "transfer_kv_all_layer",
        "transfer_kv_all_layer_direct_lf_pf",
        "transfer_kv_all_layer_lf_pf",
        "transfer_kv_all_layer_lf_ph",
        "transfer_kv_all_layer_mla",
        "transfer_kv_all_layer_mla_lf_pf",
        "transfer_kv_direct",
        "transfer_kv_per_layer",
        "transfer_kv_per_layer_direct_pf_lf",
        "transfer_kv_per_layer_mla",
        "transfer_kv_per_layer_mla_pf_lf",
        "transfer_kv_per_layer_pf_lf",
        "transfer_kv_per_layer_ph_lf",
    ),
)
del _bind_unavailable
