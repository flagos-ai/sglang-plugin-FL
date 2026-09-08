# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for CUDA adapters that delegate to sgl-kernel."""

from types import SimpleNamespace
import sys

import torch


def test_silu_and_mul_uses_sgl_kernel_input_first_contract(monkeypatch):
    from sglang_fl.dispatch.backends.vendor.cuda.impl.activation import (
        silu_and_mul_cuda,
    )

    observed = {}

    def silu_and_mul(input_tensor, out):
        observed["input"] = input_tensor
        observed["out"] = out
        out.copy_(input_tensor[..., :4])

    monkeypatch.setitem(
        sys.modules, "sgl_kernel", SimpleNamespace(silu_and_mul=silu_and_mul)
    )
    value = torch.arange(16, dtype=torch.float32).reshape(2, 8)

    result = silu_and_mul_cuda(None, value)

    assert observed["input"] is value
    assert observed["out"] is result
    assert result.shape == (2, 4)


def test_rms_norm_uses_current_sgl_kernel_function_contract(monkeypatch):
    from sglang_fl.dispatch.backends.vendor.cuda.impl.normalization import rms_norm_cuda

    observed = {}

    def rmsnorm(input_tensor, weight, eps):
        observed["args"] = (input_tensor, weight, eps)
        return input_tensor + 1

    monkeypatch.setitem(
        sys.modules,
        "sgl_kernel",
        SimpleNamespace(rmsnorm=rmsnorm, fused_add_rmsnorm=lambda *args: None),
    )
    value = torch.zeros((2, 8))
    weight = torch.ones(8)
    obj = SimpleNamespace(weight=weight, variance_epsilon=1e-6)

    result = rms_norm_cuda(obj, value)

    assert observed["args"] == (value, weight, 1e-6)
    assert torch.equal(result, torch.ones_like(value))


def test_rotary_adapter_reconstructs_sgl_kernel_cache_contract(monkeypatch):
    from sglang_fl.dispatch.backends.vendor.cuda.impl.rotary import (
        rotary_embedding_cuda,
    )

    observed = {}

    def rotary_embedding(positions, query, key, head_size, cache, is_neox):
        observed["args"] = (positions, query, key, head_size, cache, is_neox)

    monkeypatch.setitem(
        sys.modules,
        "sgl_kernel",
        SimpleNamespace(rotary_embedding=rotary_embedding),
    )
    positions = torch.tensor([0, 1])
    query = torch.zeros((2, 3, 8))
    key = torch.zeros((2, 1, 8))
    cos = torch.ones((16, 4))
    sin = torch.full((16, 4), 2.0)

    output_query, output_key = rotary_embedding_cuda(
        None,
        query,
        key,
        cos,
        sin,
        positions,
        rotary_interleaved=False,
    )

    args = observed["args"]
    assert args[:4] == (positions, query, key, 8)
    assert torch.equal(args[4], torch.cat((cos, sin), dim=-1))
    assert args[5] is True
    assert output_query is query
    assert output_key is key
