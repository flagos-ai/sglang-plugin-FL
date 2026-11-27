# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TXDA GemmaRMSNorm fp32 weight cache follows SGLang 0.5.18 weight reloads (CPU only)."""

import pytest
import torch

layernorm = pytest.importorskip("sglang.srt.layers.layernorm")

from sglang_fl.dispatch.backends.vendor.tsingmicro.impl import (  # noqa: E402
    gemma_rms_norm as g,
)


def _load(norm, values):
    # SGLang's own loader: copies into weight.data and refreshes gemma_weight in place.
    norm.weight.weight_loader(norm.weight, torch.tensor(values))


def test_cache_is_reused_until_weights_are_reloaded() -> None:
    norm = layernorm.GemmaRMSNorm(4)
    _load(norm, [0.0, 1.0, 2.0, 3.0])
    first = g._gemma_weight_fp32(norm)
    assert torch.equal(first, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert g._gemma_weight_fp32(norm) is first  # cached

    buffer_before = norm.gemma_weight
    _load(norm, [5.0, 5.0, 5.0, 5.0])  # reload, e.g. update_weights_from_disk
    assert norm.gemma_weight is buffer_before  # in place: identity unchanged
    second = g._gemma_weight_fp32(norm)
    assert second is not first
    assert torch.equal(second, torch.tensor([6.0, 6.0, 6.0, 6.0]))


def test_module_without_gemma_weight_rebuilds_every_call() -> None:
    class Plain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))

    m = Plain()
    a, b = g._gemma_weight_fp32(m), g._gemma_weight_fp32(m)
    assert a is not b and torch.equal(a, torch.ones(2))
