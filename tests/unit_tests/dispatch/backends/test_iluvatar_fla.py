# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""PR132's independent GDN wrappers honor the official SGLang 0.5.18 ABI."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang_fl.dispatch.backends.vendor.iluvatar.impl import fla


def _inputs(batch=2, length=3, *, dtype=torch.float16):
    generator = torch.Generator().manual_seed(132)
    return dict(
        q=torch.randn(batch, length, 1, 4, generator=generator).to(dtype),
        k=torch.randn(batch, length, 1, 4, generator=generator).to(dtype),
        v=torch.randn(batch, length, 2, 3, generator=generator).to(dtype),
        g=torch.randn(batch, length, 2, generator=generator),
        beta=torch.randn(batch, length, 2, generator=generator),
        scale=0.5,
    )


@pytest.fixture
def native(monkeypatch):
    """Simulate the independent in-place kernel; do not import Triton on CPU."""
    calls = []

    def recurrent(*args, return_checkpoints=False):
        q, k, v, g, beta, state, indices, cu, scale, normalize = args
        calls.append(
            SimpleNamespace(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                state=state,
                indices=indices,
                cu=cu,
                scale=scale,
                normalize=normalize,
            )
        )
        state.add_(2)
        output = v.clone()
        if not return_checkpoints:
            return output
        # Match PR132's conservative allocation, which is larger than h's ABI.
        capacity = (q.shape[0] + 63) // 64 + cu.numel() - 1
        count = capacity * v.shape[-2] * v.shape[-1] * q.shape[-1]
        checkpoints = torch.arange(count, dtype=q.dtype).reshape(
            1, capacity, v.shape[-2], v.shape[-1], q.shape[-1]
        )
        calls[-1].checkpoints = checkpoints
        return output, checkpoints

    monkeypatch.setitem(
        sys.modules,
        "sglang_fl.dispatch.backends.vendor.iluvatar.impl.gdn_triton",
        SimpleNamespace(recurrent_gdn=recurrent),
    )
    return calls


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("return_state", [False, True])
def test_public_recurrent_never_mutates_initial_state(native, dtype, return_state):
    inputs = _inputs()
    initial = torch.randn(2, 2, 3, 4).to(dtype)
    before = initial.clone()

    output, final_state = fla.fused_recurrent_gated_delta_rule_iluvatar(
        **inputs, initial_state=initial, output_final_state=return_state
    )

    assert torch.equal(initial, before)
    assert output.shape == inputs["v"].shape
    assert torch.equal(output, inputs["v"])
    assert native[0].state.dtype == torch.float32
    assert native[0].state.data_ptr() != initial.data_ptr()
    if return_state:
        assert final_state is native[0].state
        assert final_state.dtype == torch.float32
        torch.testing.assert_close(final_state, before.float() + 2)
    else:
        assert final_state is None


def test_public_recurrent_without_initial_state_returns_fp32(native):
    inputs = _inputs()
    output, state = fla.fused_recurrent_gated_delta_rule_iluvatar(
        **inputs, output_final_state=True
    )
    assert output.dtype == torch.float16
    assert state.dtype == torch.float32
    assert state.shape == (2, 2, 3, 4)
    assert torch.equal(state, torch.full_like(state, 2))
    assert torch.equal(native[0].cu, torch.tensor([0, 3, 6], dtype=torch.int32))
    assert torch.equal(native[0].indices, torch.tensor([0, 1], dtype=torch.int32))


def test_public_recurrent_varlen_preserves_state_and_forwards_offsets(native):
    inputs = _inputs(batch=1, length=5)
    bounds = torch.tensor([0, 2, 5], dtype=torch.int64)
    initial = torch.randn(2, 2, 3, 4)
    before = initial.clone()
    output, final = fla.fused_recurrent_gated_delta_rule_iluvatar(
        **inputs, initial_state=initial, cu_seqlens=bounds, output_final_state=True
    )
    assert output.shape == (1, 5, 2, 3)
    assert final.shape == (2, 2, 3, 4)
    assert torch.equal(initial, before)
    assert torch.equal(final, before + 2)
    assert native[0].cu.dtype == torch.int32
    assert torch.equal(native[0].cu, bounds.int())


def test_legacy_indexed_recurrent_also_owns_its_working_pool(native):
    inputs = _inputs()
    initial = torch.randn(5, 2, 3, 4)
    before = initial.clone()
    indices = torch.tensor([3, 1], dtype=torch.int64)
    _, final = fla.fused_recurrent_gated_delta_rule_iluvatar(
        **inputs,
        initial_state=initial,
        ssm_state_indices=indices,
        output_final_state=True,
    )
    assert torch.equal(initial, before)
    assert final.dtype == torch.float32
    assert native[0].indices.dtype == torch.int32
    assert torch.equal(native[0].indices, indices.int())


def test_recurrent_rejects_rollback_before_running_a_kernel(native):
    with pytest.raises(NotImplementedError, match="rollback"):
        fla.fused_recurrent_gated_delta_rule_iluvatar(
            **_inputs(), num_accepted_tokens=torch.tensor([1, 1])
        )
    assert native == []


@pytest.mark.parametrize("with_state", [False, True])
def test_chunk_unpooled_returns_none_and_exact_dense_checkpoints(native, with_state):
    inputs = _inputs(length=65)
    state = torch.zeros(2, 2, 3, 4) if with_state else None
    output, final_state, checkpoints = fla.chunk_gated_delta_rule_iluvatar(
        **inputs, initial_state=state
    )
    # Official chunk_delta_h allocates [B,ceil(T/64),HV,V,K], even for B>1.
    assert output.shape == (2, 65, 2, 3)
    assert final_state is None
    assert checkpoints.shape == (2, 2, 2, 3, 4)
    assert native[0].checkpoints.shape == (1, 5, 2, 3, 4)
    expected = native[0].checkpoints[:, :4].reshape(2, 2, 2, 3, 4)
    assert torch.equal(checkpoints, expected)
    if with_state:
        # This ownership guarantee is our unindexed convenience extension,
        # not a guarantee made by the official indexed SGLang chunk kernel.
        assert native[0].state.data_ptr() != state.data_ptr()
        assert torch.equal(state, torch.zeros_like(state))
        assert torch.equal(native[0].state, torch.full_like(state, 2))


def test_chunk_varlen_returns_exact_sum_of_per_sequence_chunk_counts(native):
    inputs = _inputs(batch=1, length=76)
    bounds = torch.tensor([0, 5, 72, 76], dtype=torch.int64)
    slots = torch.tensor([4, -1, 1], dtype=torch.int64)
    state = torch.zeros(6, 2, 3, 4)
    output, final_state, checkpoints = fla.chunk_gated_delta_rule_iluvatar(
        **inputs, initial_state=state, initial_state_indices=slots, cu_seqlens=bounds
    )
    assert output.shape == (1, 76, 2, 3)
    assert final_state is None
    # ceil(5/64)+ceil(67/64)+ceil(4/64) = 4, never max capacity 5.
    assert checkpoints.shape == (1, 4, 2, 3, 4)
    assert native[0].checkpoints.shape == (1, 5, 2, 3, 4)
    assert torch.equal(checkpoints, native[0].checkpoints[:, :4])
    assert native[0].state is state
    assert torch.equal(native[0].indices, slots.int())


def test_head_first_chunk_restores_output_but_not_checkpoint_axes(native):
    inputs = _inputs(length=65)
    for name in ("q", "k", "v", "g", "beta"):
        inputs[name] = inputs[name].transpose(1, 2)
    output, final_state, checkpoints = fla.chunk_gated_delta_rule_iluvatar(
        **inputs, head_first=True
    )
    assert output.shape == inputs["v"].shape
    assert checkpoints.shape == (2, 2, 2, 3, 4)
    assert final_state is None


def test_chunk_empty_dense_sequence_has_zero_public_checkpoints(native):
    output, final_state, checkpoints = fla.chunk_gated_delta_rule_iluvatar(
        **_inputs(length=0)
    )
    assert output.shape == (2, 0, 2, 3)
    assert final_state is None
    assert checkpoints.shape == (2, 0, 2, 3, 4)
    assert torch.equal(native[0].cu, torch.zeros(3, dtype=torch.int32))


@pytest.mark.parametrize(
    "buffer",
    [torch.zeros(1, 1, 2, 3, 4), torch.zeros(1, 5, 2, 4, 3)],
)
def test_checkpoint_capacity_and_layout_are_validated(buffer):
    with pytest.raises(RuntimeError, match="checkpoint"):
        fla._chunk_checkpoints(buffer, (2, 65, 2, 3), 4, None)


def test_none_scale_uses_the_upstream_default(native):
    inputs = _inputs()
    inputs["scale"] = None
    fla.chunk_gated_delta_rule_iluvatar(**inputs)
    assert native[-1].scale == 0.5
    fla.fused_recurrent_gated_delta_rule_iluvatar(**inputs)
    assert native[-1].scale == 0.5


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"initial_state": torch.zeros(3, 2, 3, 4)}, "must have shape"),
        ({"initial_state": torch.zeros(2, 2, 4, 3)}, "trailing shape"),
        ({"ssm_state_indices": torch.tensor([0, 1])}, "require initial_state"),
        ({"cu_seqlens": torch.tensor([0, 3, 6])}, "batch size 1"),
    ],
)
def test_invalid_recurrent_metadata_is_rejected_before_kernel(native, extra, match):
    with pytest.raises(ValueError, match=match):
        fla.fused_recurrent_gated_delta_rule_iluvatar(**_inputs(), **extra)
    assert native == []


def test_invalid_chunk_slot_count_is_rejected_before_kernel(native):
    with pytest.raises(ValueError, match="one entry per sequence"):
        fla.chunk_gated_delta_rule_iluvatar(
            **_inputs(),
            initial_state=torch.zeros(3, 2, 3, 4),
            initial_state_indices=torch.tensor([0]),
        )
    assert native == []


def test_recurrent_default_omits_final_state(native):
    output, final_state = fla.fused_recurrent_gated_delta_rule_iluvatar(**_inputs())
    assert output.shape == (2, 3, 2, 3)
    assert final_state is None


def test_backend_recurrent_forwards_false_default(monkeypatch):
    from sglang_fl.dispatch.backends.vendor.iluvatar.iluvatar import IluvatarBackend

    calls = []
    sentinel = object()

    def implementation(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(
        fla, "fused_recurrent_gated_delta_rule_iluvatar", implementation
    )
    result = IluvatarBackend().fused_recurrent_gated_delta_rule(**_inputs())
    assert result is sentinel
    assert calls[0][7] is False
