# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from sglang_fl.dispatch.backends.vendor.mthreads.runtime_config import (
    apply_musa_runtime_defaults,
)


@pytest.mark.parametrize(
    "arch,nodes,spec,disabled",
    [
        ("Qwen3_5MoeForConditionalGeneration", 2, None, False),
        ("Qwen3_5MoeForConditionalGeneration", 1, None, False),
        ("Qwen3_5ForConditionalGeneration", 2, None, False),
        ("Qwen3_5ForConditionalGeneration", 1, "EAGLE", True),
        ("Qwen3_5MoeForConditionalGeneration", 1, "EAGLE", True),
        ("LlamaForCausalLM", 2, "EAGLE", False),
    ],
)
def test_runtime_fallback_scope(arch, nodes, spec, disabled):
    args = SimpleNamespace(
        nnodes=nodes,
        speculative_algorithm=spec,
        disable_overlap_schedule=False,
        disable_radix_cache=bool(spec),
        disable_cuda_graph=False,
        mamba_radix_cache_strategy="auto",
        page_size=64,
        max_running_requests=48,
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[arch])
        ),
    )
    apply_musa_runtime_defaults(args)
    assert args.disable_overlap_schedule is disabled
    assert not args.disable_cuda_graph
    assert args.max_running_requests == 48
    assert (args.mamba_radix_cache_strategy, args.page_size) == ("auto", 64)


def test_legacy_args_do_not_require_new_api():
    apply_musa_runtime_defaults(SimpleNamespace())
