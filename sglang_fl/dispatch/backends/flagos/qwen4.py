# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Qwen3.8 numerical contracts supplied by FlagGems-sglang.

HC/gate/norm return fresh tensors. KV/compression stores mutate only cache
rows selected by slots; PLE convolution mutates output and selected state.
Sparse attention mutates caller-owned output/workspace when supplied.
"""

from sglang_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl

OP_NAMES = (
    "fla_layernorm_gated",
    "qwen4_grouped_gemma_rmsnorm",
    "qwen4_hc_gate_reduce",
    "qwen4_hc_inject_combine",
    "qwen4_hc_combine_norm",
    "qwen4_hc_scale_silu",
    "qwen4_qsa_all_blocks",
    "qwen4_qsa_topk",
    "qwen4_qsa_partial_rope",
    "qsa_mqa_paged",
    "qsa_store_kv_cache_rows",
    "qsa_compress_norm_mrope_store_groups",
    "qsa_sparse_paged_attention",
    "ple_gate_norm",
    "ple_short_conv_",
    "ple_prefill_short_conv_",
)


def register_builtins(registry):
    import flaggems_sglang

    for name in OP_NAMES:
        function = flaggems_sglang.get_op(name)
        if function is not None:
            registry.register_impl(
                OpImpl(
                    op_name=name,
                    impl_id="default.flagos",
                    kind=BackendImplKind.DEFAULT,
                    fn=function,
                    priority=BackendPriority.DEFAULT,
                )
            )
