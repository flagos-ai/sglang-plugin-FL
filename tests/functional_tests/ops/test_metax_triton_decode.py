# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Functional (on-device) check for the MetaX triton decode kv-split policy.

``patches/triton_attention.py`` caps the decode ``max_kv_splits`` for MetaX.
With a fixed split tile (deterministic inference) upstream sets
``num_kv_splits = ceil(seq_len / tile)`` without clamping it, so a cap below
``ceil(context_len / tile)`` drops the tail of the KV cache. This runs the
patched decode kernels with the bound the MetaX policy picks and compares the
output against a float32 torch reference.

Skips unless the MetaX backend is available.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from sglang_fl.dispatch.backends.vendor.metax.metax import MetaxBackend

pytestmark = [pytest.mark.functional, pytest.mark.gpu]


def _metax_available() -> bool:
    try:
        return MetaxBackend().is_available()
    except Exception:
        return False


pytestmark.append(
    pytest.mark.skipif(not _metax_available(), reason="MetaX backend unavailable")
)

SPLIT_TILE = 256  # SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE default
CONTEXT_LEN = 32768


def _reference(q, k, v, kv_indptr, sm_scale):
    """float32 decode attention: q (B, Hq, D), k/v (T, Hkv, D)."""
    out = torch.empty(q.shape[0], q.shape[1], v.shape[-1], dtype=torch.float32)
    group = q.shape[1] // k.shape[1]
    for b in range(q.shape[0]):
        lo, hi = int(kv_indptr[b]), int(kv_indptr[b + 1])
        kb = k[lo:hi].float().repeat_interleave(group, dim=1)  # (L, Hq, D)
        vb = v[lo:hi].float().repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,lhd->hl", q[b].float(), kb) * sm_scale
        out[b] = torch.einsum("hl,lhd->hd", scores.softmax(-1), vb)
    return out


def _decode(q, k, v, kv_indptr, num_kv_splits, max_kv_splits, sm_scale):
    from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd

    batch, heads, dv = q.shape[0], q.shape[1], v.shape[-1]
    o = torch.empty(batch, heads, dv, dtype=q.dtype, device=q.device)
    attn_logits = torch.empty(
        batch, heads, max_kv_splits, dv, dtype=torch.float32, device=q.device
    )
    attn_lse = torch.empty(
        batch, heads, max_kv_splits, dtype=torch.float32, device=q.device
    )
    kv_indices = torch.arange(k.shape[0], dtype=torch.int64, device=q.device)
    decode_attention_fwd(
        q, k, v, o, kv_indptr, kv_indices, attn_logits, attn_lse,
        num_kv_splits, max_kv_splits, sm_scale, 1.0, 1.0,
    )  # fmt: skip
    return o.float().cpu()


@pytest.mark.parametrize("extend_lens", [None, [12], [8192]])
def test_deterministic_long_context_decode_covers_all_tokens(extend_lens):
    from sglang_fl.dispatch.backends.vendor.metax.patches import triton_attention

    # Exercise the MetaX launch overrides (decode BLOCK_N / num_stages).
    triton_attention.patch_triton_attention()

    torch.manual_seed(0)
    device = "cuda"
    seq_lens = [8192, 1000, 300]
    heads_q, heads_kv, head_dim = 8, 2, 128
    sm_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(len(seq_lens), heads_q, head_dim, dtype=torch.bfloat16)
    k = torch.randn(sum(seq_lens), heads_kv, head_dim, dtype=torch.bfloat16)
    v = torch.randn(sum(seq_lens), heads_kv, head_dim, dtype=torch.bfloat16)
    kv_indptr = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()])
    expected = _reference(q, k, v, kv_indptr, sm_scale)

    # What upstream TritonAttnBackend computes with --enable-deterministic-inference.
    backend = SimpleNamespace(
        split_tile_size=SPLIT_TILE,
        max_kv_splits=(CONTEXT_LEN + SPLIT_TILE - 1) // SPLIT_TILE,
    )
    max_kv_splits = triton_attention.metax_max_kv_splits(backend, extend_lens)
    num_kv_splits = torch.tensor(
        [(n + SPLIT_TILE - 1) // SPLIT_TILE for n in seq_lens], dtype=torch.int32
    )

    args = (
        q.to(device), k.to(device), v.to(device), kv_indptr.to(device, torch.int32),
        num_kv_splits.to(device),
    )  # fmt: skip
    out = _decode(*args, max_kv_splits, sm_scale)
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)

    # The fixed MetaX bound (16) covers only 16 * 256 tokens of the 8192-token
    # sequence; make sure this test would have caught it.
    truncated = _decode(*args, 16, sm_scale)
    rel_err = (truncated[0] - expected[0]).norm() / expected[0].norm()
    assert rel_err > 0.3, rel_err
