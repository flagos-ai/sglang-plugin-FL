# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for ``impl/fla.py`` (Gated-DeltaNet on klx).

Both the prefill (chunk) and decode (packed, T=1) entries are backed by
``torch.ops.sgl_kernel.klx_gated_delta_net``. The kernel is stubbed with a
deterministic transform, so the tests pin what the wrapper owns:

  * SGLang stores SSM state as ``[.., HV, V, K]`` while klx wants
    ``[.., HV, K, V]`` — the double transpose around the kernel call,
  * gather/scatter around the full ``ssm_states`` cache (in-place update),
  * int32 CPU ``cu_seqlens`` and fp16/fp32 gate dtype rules,
  * packed-decode QKV split + softplus gating reproduced in torch,
  * the per-chunk state checkpoints consumed by SGLang's radix mamba tracking.
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from sglang_fl.dispatch.backends.vendor.kunlunxin.impl import fla


class _ChunkKernel:
    """Deterministic stand-in for ``klx_gated_delta_net``."""

    def __init__(self, chunk_states=None):
        self.calls = []
        self.chunk_states = chunk_states

    def __call__(
        self,
        q,
        k,
        v,
        g,
        beta,
        scale,
        h0,
        output_final_state,
        cu_seqlens,
        head_first,
        use_qk_l2norm_in_kernel,
    ):
        self.calls.append(
            SimpleNamespace(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                h0=h0,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                head_first=head_first,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
        )
        out = v * float(scale)
        final_state = h0 + 1
        if self.chunk_states is None:
            return [out, final_state]
        return [out, final_state, self.chunk_states]


@pytest.fixture
def kernel(monkeypatch):
    def _install(chunk_states=None):
        stub = _ChunkKernel(chunk_states)
        monkeypatch.setattr(
            torch.ops,
            "sgl_kernel",
            SimpleNamespace(klx_gated_delta_net=stub),
            raising=False,
        )
        monkeypatch.setitem(sys.modules, "sgl_kernel", types.ModuleType("sgl_kernel"))
        return stub

    return _install


def _prefill_inputs(dtype=torch.float16, tokens=6, heads=2, head_dim=4, v_dim=4, lines=4):
    torch.manual_seed(0)
    q = torch.randn(1, tokens, heads, head_dim, dtype=dtype)
    k = torch.randn(1, tokens, heads, head_dim, dtype=dtype)
    v = torch.randn(1, tokens, heads, v_dim, dtype=dtype)
    g = torch.randn(1, tokens, heads, dtype=torch.float32)
    beta = torch.randn(1, tokens, heads, dtype=torch.float32)
    states = torch.randn(lines, heads, v_dim, head_dim, dtype=dtype)
    return SimpleNamespace(
        q=q, k=k, v=v, g=g, beta=beta, states=states, cu=torch.tensor([0, tokens])
    )


def test_prefill_feeds_klx_layout_and_updates_the_state_cache(kernel):
    stub = kernel()
    inputs = _prefill_inputs()
    before = inputs.states.clone()  # the cache is updated in place
    indices = torch.tensor([2], dtype=torch.int32)

    out, final, chunk_h = fla.chunk_gated_delta_rule_kunlunxin(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.g,
        inputs.beta,
        2.0,
        inputs.states,
        indices,
        inputs.cu,
        False,
        True,
    )

    call = stub.calls[-1]
    # fp16 activations require fp16 gates; cu_seqlens must be int32 on the CPU.
    assert call.g.dtype is torch.float16
    assert call.beta.dtype is torch.float16
    assert call.cu_seqlens.dtype is torch.int32
    assert call.cu_seqlens.device.type == "cpu"
    assert call.output_final_state is True
    assert call.head_first is False
    assert call.use_qk_l2norm_in_kernel is True
    # SGLang [N, HV, V, K] -> klx [N, HV, K, V]
    assert call.h0.shape == (1, 2, 4, 4)
    assert torch.equal(call.h0[0], before[2].transpose(-1, -2))
    assert out.shape == inputs.v.shape
    assert final is None
    assert chunk_h is None
    # Final state is scattered back in place, transposed to SGLang's layout.
    assert torch.equal(inputs.states[2], (call.h0[0] + 1).transpose(-1, -2))
    for row in (0, 1, 3):
        assert torch.equal(inputs.states[row], before[row])


def test_prefill_uses_fp32_gates_for_bf16_activations(kernel):
    stub = kernel()
    inputs = _prefill_inputs(dtype=torch.bfloat16)

    fla.chunk_gated_delta_rule_kunlunxin(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.g,
        inputs.beta,
        1.0,
        inputs.states,
        torch.tensor([0], dtype=torch.int32),
        inputs.cu,
    )

    call = stub.calls[-1]
    # klx rejects bf16 gates: non-fp16 activations run with fp32 gates/state.
    assert call.g.dtype is torch.float32
    assert call.beta.dtype is torch.float32
    assert call.h0.dtype is torch.float32
    assert inputs.states.dtype is torch.bfloat16


def test_prefill_without_initial_state_starts_from_zeros(kernel):
    stub = kernel()
    inputs = _prefill_inputs()

    fla.chunk_gated_delta_rule_kunlunxin(
        inputs.q, inputs.k, inputs.v, inputs.g, inputs.beta, 1.0,
        initial_state=None,
        initial_state_indices=None,
        cu_seqlens=inputs.cu,
    )

    call = stub.calls[-1]
    assert call.h0.shape == (1, 2, 4, 4)
    assert torch.equal(call.h0, torch.zeros_like(call.h0))


def test_prefill_accepts_int32_cu_seqlens(kernel):
    """SGLang may pass int32; the kernel contract needs int32 on the CPU."""
    stub = kernel()
    inputs = _prefill_inputs()
    inputs.cu = inputs.cu.to(torch.int32)

    fla.chunk_gated_delta_rule_kunlunxin(
        inputs.q, inputs.k, inputs.v, inputs.g, inputs.beta, 1.0,
        None, None, inputs.cu,
    )

    recorded = stub.calls[-1].cu_seqlens
    assert recorded.dtype is torch.int32
    assert recorded.device.type == "cpu"


def test_prefill_returns_chunk_state_checkpoints_for_radix_tracking(kernel, monkeypatch):
    monkeypatch.setattr(fla, "_fla_chunk_size", lambda: 64)
    monkeypatch.setattr(fla, "_mamba_cache_chunk_size", lambda fallback: 64)

    heads, head_dim, v_dim = 2, 4, 4
    chunk_states = torch.arange(3 * heads * head_dim * v_dim, dtype=torch.float32)
    chunk_states = chunk_states.reshape(3, heads, head_dim, v_dim)
    stub = kernel(chunk_states)
    inputs = _prefill_inputs(tokens=6, heads=heads, head_dim=head_dim, v_dim=v_dim)
    # 1 chunk for the first request, 2 for the second (ceil(128 / 64)).
    inputs.cu = torch.tensor([0, 64, 192])

    _, _, chunk_h = fla.chunk_gated_delta_rule_kunlunxin(
        inputs.q, inputs.k, inputs.v, inputs.g, inputs.beta, 1.0,
        None, None, inputs.cu,
    )

    # Request 0 keeps the incoming state; request 1 keeps one checkpoint per
    # cache chunk (its read index clamps to the last available chunk row).
    expected = (
        chunk_states.index_select(0, torch.tensor([0, 2, 2]))
        .transpose(-1, -2)
        .unsqueeze(0)
        .to(torch.float16)
    )
    assert chunk_h.shape == (1, 3, heads, v_dim, head_dim)
    assert torch.equal(chunk_h, expected)


def test_state_checkpoint_stride_follows_the_cache_chunk(monkeypatch):
    monkeypatch.setattr(fla, "_fla_chunk_size", lambda: 64)
    monkeypatch.setattr(fla, "_mamba_cache_chunk_size", lambda fallback: 128)

    heads, head_dim, v_dim = 1, 2, 2
    rows = 4
    hk = torch.arange(rows * heads * head_dim * v_dim, dtype=torch.float32).reshape(
        rows, heads, head_dim, v_dim
    )
    cu = torch.tensor([0, 200], dtype=torch.int32)

    selected = fla._select_state_checkpoints(hk, cu, torch.float32)

    expected = hk.index_select(0, torch.tensor([2, 3])).transpose(-1, -2).unsqueeze(0)
    assert torch.equal(selected, expected)


def test_state_checkpoints_are_empty_for_an_empty_batch(monkeypatch):
    monkeypatch.setattr(fla, "_fla_chunk_size", lambda: 64)
    monkeypatch.setattr(fla, "_mamba_cache_chunk_size", lambda fallback: 64)

    hk = torch.empty(0, 2, 4, 4)
    selected = fla._select_state_checkpoints(
        hk, torch.tensor([0], dtype=torch.int32), torch.float32
    )

    assert selected.shape == (1, 0, 2, 4, 4)
    assert selected.numel() == 0


# ─── packed decode ──────────────────────────────────────────────────────────


def _decode_inputs(dtype=torch.float32, batch=2, heads=2, head_dim=4, v_dim=4, lines=3):
    torch.manual_seed(1)
    qkv_dim = 2 * heads * head_dim + heads * v_dim
    mixed_qkv = torch.randn(batch, qkv_dim, dtype=dtype)
    a = torch.randn(batch, heads, dtype=torch.float32)
    b = torch.randn(batch, heads, dtype=torch.float32)
    a_log = torch.randn(heads, dtype=torch.float32)
    dt_bias = torch.randn(heads, dtype=torch.float32)
    states = torch.randn(lines, heads, v_dim, head_dim, dtype=dtype)
    out = torch.zeros(batch, 1, heads, v_dim, dtype=dtype)
    return SimpleNamespace(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        a_log=a_log,
        dt_bias=dt_bias,
        states=states,
        out=out,
        indices=torch.tensor([2, 0], dtype=torch.int32),
        heads=heads,
        head_dim=head_dim,
        v_dim=v_dim,
    )


def test_decode_reproduces_split_and_softplus_gating(kernel):
    stub = kernel()
    inputs = _decode_inputs()
    states_before = inputs.states.clone()  # the cache is updated in place
    heads, head_dim, v_dim = inputs.heads, inputs.head_dim, inputs.v_dim
    expected_g = -inputs.a_log.exp().unsqueeze(0) * torch.nn.functional.softplus(
        inputs.a + inputs.dt_bias
    )
    expected_beta = torch.sigmoid(inputs.b)

    out, returned_state = fla.fused_recurrent_gated_delta_rule_packed_decode_kunlunxin(
        inputs.mixed_qkv,
        inputs.a,
        inputs.b,
        inputs.a_log,
        inputs.dt_bias,
        1.5,
        inputs.states,
        inputs.out,
        inputs.indices,
        True,
    )

    call = stub.calls[-1]
    split = heads * head_dim
    assert call.q.shape == (2, 1, heads, head_dim)
    assert torch.equal(call.q, inputs.mixed_qkv[:, :split].reshape(2, 1, heads, head_dim))
    assert torch.equal(
        call.k, inputs.mixed_qkv[:, split : 2 * split].reshape(2, 1, heads, head_dim)
    )
    assert torch.equal(
        call.v, inputs.mixed_qkv[:, 2 * split :].reshape(2, 1, heads, v_dim)
    )
    assert torch.allclose(call.g, expected_g.reshape(2, 1, heads), atol=1e-6)
    assert torch.allclose(call.beta, expected_beta.reshape(2, 1, heads), atol=1e-6)
    assert call.g.dtype is torch.float32
    assert call.cu_seqlens.tolist() == [0, 1, 2]
    assert call.cu_seqlens.dtype is torch.int32
    assert call.head_first is False
    assert call.output_final_state is True
    assert call.use_qk_l2norm_in_kernel is True
    # One token per sequence, states transposed into klx layout.
    assert call.h0.shape == (2, heads, head_dim, v_dim)
    assert torch.equal(call.h0[0], states_before[2].transpose(-1, -2))
    assert torch.equal(call.h0[1], states_before[0].transpose(-1, -2))
    # Decode result is written into the caller-provided buffer.
    assert out is inputs.out
    assert torch.equal(out, call.v * 1.5)
    assert returned_state is inputs.states
    assert torch.equal(inputs.states[2], (call.h0[0] + 1).transpose(-1, -2))
    assert torch.equal(inputs.states[0], (call.h0[1] + 1).transpose(-1, -2))


def test_decode_softplus_switches_to_linear_above_the_threshold(kernel):
    stub = kernel()
    inputs = _decode_inputs()
    inputs.a = torch.full_like(inputs.a, 30.0)
    inputs.dt_bias = torch.zeros_like(inputs.dt_bias)
    inputs.a_log = torch.zeros_like(inputs.a_log)

    fla.fused_recurrent_gated_delta_rule_packed_decode_kunlunxin(
        inputs.mixed_qkv, inputs.a, inputs.b, inputs.a_log, inputs.dt_bias, 1.0,
        inputs.states, inputs.out, inputs.indices,
    )

    # softplus(x) == x for x > 20 (SGLang's SOFTPLUS_THRESHOLD), and A_log == 0.
    assert torch.allclose(stub.calls[-1].g, torch.full((2, 1, 2), -30.0))


def test_decode_flattens_a_stray_token_axis(kernel):
    stub = kernel()
    inputs = _decode_inputs()
    inputs.mixed_qkv = inputs.mixed_qkv.unsqueeze(1)

    fla.fused_recurrent_gated_delta_rule_packed_decode_kunlunxin(
        inputs.mixed_qkv, inputs.a, inputs.b, inputs.a_log, inputs.dt_bias, 1.0,
        inputs.states, inputs.out, inputs.indices,
    )

    assert stub.calls[-1].q.shape[0] == 2


def test_decode_uses_fp16_state_for_fp16_activations(kernel):
    stub = kernel()
    inputs = _decode_inputs(dtype=torch.float16)

    fla.fused_recurrent_gated_delta_rule_packed_decode_kunlunxin(
        inputs.mixed_qkv, inputs.a, inputs.b, inputs.a_log, inputs.dt_bias, 1.0,
        inputs.states, inputs.out, inputs.indices,
    )

    call = stub.calls[-1]
    assert call.g.dtype is torch.float16
    assert call.beta.dtype is torch.float16
    assert call.h0.dtype is torch.float16
    assert inputs.out.dtype is torch.float16
