# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for ``patches/gdn_fused_qkv_split.py``.

SGLang 0.5.18 selects the Triton ``fused_qkv_split_gdn_prefill`` whenever
``qkv_dim <= 8192``. Under TP for Qwen3.6-27B that is 5120 (TP2) / 2560 (TP4),
and the Triton split corrupts K/V on strided post-conv views on Kunlun. The
patch therefore (a) rebinds the split to ``torch.split`` and (b) closes the
call-site gate, so a stale import cannot reselect the fused path.
"""

import importlib
from types import SimpleNamespace

import torch

from sglang_fl.dispatch.backends.vendor.kunlunxin.patches import (
    gdn_fused_qkv_split as patch_module,
)

_GDN_PARAMS = SimpleNamespace(
    num_q_heads=4, num_k_heads=2, num_v_heads=2, head_q=8, head_k=8, head_v=8
)


def _mixed_qkv(seq_len=5):
    dim = (
        _GDN_PARAMS.num_q_heads * _GDN_PARAMS.head_q
        + _GDN_PARAMS.num_k_heads * _GDN_PARAMS.head_k
        + _GDN_PARAMS.num_v_heads * _GDN_PARAMS.head_v
    )
    return torch.arange(seq_len * dim, dtype=torch.float32).reshape(seq_len, dim)


def test_split_matches_the_contiguous_torch_fallback():
    mixed_qkv = _mixed_qkv()
    params = _GDN_PARAMS

    query, key, value = patch_module._torch_qkv_split_gdn_prefill(
        mixed_qkv,
        params.num_q_heads,
        params.num_k_heads,
        params.num_v_heads,
        params.head_q,
        params.head_k,
        params.head_v,
    )

    q_ref, k_ref, v_ref = torch.split(
        mixed_qkv,
        [
            params.num_q_heads * params.head_q,
            params.num_k_heads * params.head_k,
            params.num_v_heads * params.head_v,
        ],
        dim=-1,
    )
    seq_len = mixed_qkv.shape[0]
    assert query.shape == (1, seq_len, params.num_q_heads, params.head_q)
    assert key.shape == (1, seq_len, params.num_k_heads, params.head_k)
    assert value.shape == (1, seq_len, params.num_v_heads, params.head_v)
    assert torch.equal(query, q_ref.view(1, seq_len, params.num_q_heads, params.head_q))
    assert torch.equal(key, k_ref.view(1, seq_len, params.num_k_heads, params.head_k))
    assert torch.equal(value, v_ref.view(1, seq_len, params.num_v_heads, params.head_v))


def test_split_preserves_values_on_a_transposed_post_conv_view():
    """The fused Triton kernel corrupted K/V on strided post-conv views.

    ``gdn_backend`` feeds ``causal_conv1d_fn(...).transpose(0, 1)[:seq_len]``,
    i.e. a transposed (non-contiguous) tensor.
    """
    params = _GDN_PARAMS
    dim = (
        params.num_q_heads * params.head_q
        + params.num_k_heads * params.head_k
        + params.num_v_heads * params.head_v
    )
    seq_len = 6
    transposed = torch.arange(dim * seq_len, dtype=torch.float32).reshape(dim, seq_len).t()
    assert not transposed.is_contiguous()

    actual = patch_module._torch_qkv_split_gdn_prefill(
        transposed,
        params.num_q_heads,
        params.num_k_heads,
        params.num_v_heads,
        params.head_q,
        params.head_k,
        params.head_v,
    )
    expected = _torch_split(transposed.contiguous())
    for got, want in zip(actual, expected):
        assert torch.equal(got, want)


def _torch_split(mixed_qkv):
    params = _GDN_PARAMS
    seq_len = mixed_qkv.shape[0]
    q, k, v = torch.split(
        mixed_qkv,
        [
            params.num_q_heads * params.head_q,
            params.num_k_heads * params.head_k,
            params.num_v_heads * params.head_v,
        ],
        dim=-1,
    )
    return (
        q.view(1, seq_len, params.num_q_heads, params.head_q),
        k.view(1, seq_len, params.num_k_heads, params.head_k),
        v.view(1, seq_len, params.num_v_heads, params.head_v),
    )
