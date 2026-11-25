# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TXDA GDN verify conv: per-token updates on a private history equal one multi-token
verify call (CPU; the single-token kernel is replaced by a torch decode update)."""

import pytest
import torch

pytest.importorskip("sglang.srt.layers.attention.linear.gdn_backend")

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches import (  # noqa: E402
    gdn_verify_conv as gvc,
)


def _decode_update(x, conv_state, weight, bias=None, activation=None, **kwargs):
    """Torch stand-in for SGLang's single-token causal_conv1d_update (no indices):
    window = [state, x]; out = silu(sum(window * weight)); state shifts by one."""
    assert not kwargs, kwargs
    window = torch.cat([conv_state, x.unsqueeze(-1)], dim=-1)  # (B, dim, width)
    out = (window.float() * weight.float()).sum(-1)
    if bias is not None:
        out = out + bias.float()
    out = out * torch.sigmoid(out)
    conv_state.copy_(window[..., 1:])
    return out.to(x.dtype)


def _reference(x, conv_state, weight, cs_idx, inter, inter_idx):
    """Multi-token verify: outputs per draft token, window after each token saved,
    persistent conv_state untouched."""
    B, dim, T = x.shape
    out = torch.empty_like(x)
    for b in range(B):
        if int(cs_idx[b]) == -1:
            continue
        hist = conv_state[int(cs_idx[b])].clone()
        for t in range(T):
            window = torch.cat([hist, x[b, :, t : t + 1]], dim=-1)
            acc = (window.float() * weight.float()).sum(-1)
            out[b, :, t] = (acc * torch.sigmoid(acc)).to(x.dtype)
            hist = window[..., 1:]
            inter[int(inter_idx[b]), t] = hist
    return out


@pytest.fixture
def wrapped(monkeypatch):
    monkeypatch.setattr(gvc, "_on_txda", lambda x: True)
    return gvc._make_per_token(_decode_update)


def _case(pad):
    g = torch.Generator().manual_seed(0)
    B, dim, width, T, cache = 3, 16, 4, 4, 6
    x = torch.randn(B, dim, T, generator=g)
    conv_state = torch.randn(cache, dim, width - 1, generator=g)
    weight = torch.randn(dim, width, generator=g)
    cs_idx = torch.tensor([4, -1 if pad else 2, 0])
    inter_idx = torch.tensor([1, 3, 0])
    return x, conv_state, weight, cs_idx, inter_idx, cache, T


@pytest.mark.parametrize("pad", [False, True])
def test_per_token_matches_multi_token_verify(wrapped, pad) -> None:
    x, conv_state, weight, cs_idx, inter_idx, cache, T = _case(pad)
    inter = torch.zeros(cache, T, x.shape[1], weight.shape[1] - 1)
    inter_ref = inter.clone()
    state_before = conv_state.clone()

    out = wrapped(
        x, conv_state, weight, None, "silu",
        conv_state_indices=cs_idx,
        intermediate_conv_window=inter,
        intermediate_state_indices=inter_idx,
    )
    want = _reference(x, state_before, weight, cs_idx, inter_ref, inter_idx)

    assert out.shape == x.shape
    assert torch.equal(conv_state, state_before)  # persistent state untouched
    for b in range(x.shape[0]):
        if int(cs_idx[b]) == -1:
            continue  # padding row: output is unused
        assert torch.allclose(out[b], want[b], atol=1e-6)
        assert torch.allclose(inter[int(inter_idx[b])], inter_ref[int(inter_idx[b])], atol=1e-6)
    if pad:
        # padding writes go to the discard row (last row), not to request 1's row
        assert torch.equal(inter[int(inter_idx[1])], torch.zeros_like(inter[0]))


def test_other_calls_go_to_upstream(monkeypatch) -> None:
    calls = []

    def upstream(*args, **kwargs):
        calls.append(kwargs)
        return "upstream"

    monkeypatch.setattr(gvc, "_on_txda", lambda x: True)
    fn = gvc._make_per_token(upstream)
    x = torch.zeros(1, 4, 2)
    # tree verify (retrieve_* given) and decode (no intermediate window)
    assert fn(x, None, None, conv_state_indices=torch.zeros(1), intermediate_conv_window=torch.zeros(1),
              intermediate_state_indices=torch.zeros(1), retrieve_next_token=torch.zeros(1)) == "upstream"
    assert fn(x[..., 0], None, None) == "upstream"
    monkeypatch.setattr(gvc, "_on_txda", lambda x: False)
    assert fn(x, None, None, conv_state_indices=torch.zeros(1), intermediate_conv_window=torch.zeros(1),
              intermediate_state_indices=torch.zeros(1)) == "upstream"
    assert len(calls) == 3
