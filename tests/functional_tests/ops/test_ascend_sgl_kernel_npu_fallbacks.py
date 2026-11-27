# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Numerics for the Ascend sgl_kernel_npu fallbacks.

The CPU tests prove the plain-PyTorch implementations against scalar
references without hardware. The device tests prove the Triton-delegated
GDN fallbacks run and shape-match on a real NPU; they skip everywhere else
(the CI functional job on 910C is the gate).
"""

from __future__ import annotations

from itertools import accumulate
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).parents[3]

_fallbacks = pytest.importorskip(
    "sglang_fl.dispatch.backends.vendor.ascend.sgl_kernel_npu_fallbacks"
)


def _npu_available() -> bool:
    try:
        import torch  # noqa: F811

        return bool(torch.npu.is_available()) if hasattr(torch, "npu") else False
    except (ImportError, RuntimeError):
        return False


# ---------------------------------------------------------------------------
# l1_norm (plain PyTorch)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_l1_norm_normalizes_each_row(dtype: torch.dtype) -> None:
    scores = torch.tensor([[1.0, 3.0, 0.0], [2.0, 2.0, 4.0], [-1.0, 1.0, 1.0]])
    result = _fallbacks.l1_norm(scores.to(dtype))
    expected = scores / scores.abs().sum(dim=-1, keepdim=True)
    assert result.dtype == dtype
    assert torch.allclose(result.float(), expected, atol=2e-2)


def test_l1_norm_zero_row_does_not_divide_by_zero_into_nan_grads() -> None:
    # A fully zero row stays finite (0 / 0); the vendor kernel produces the
    # same, and MoE routing masks such rows downstream.
    result = _fallbacks.l1_norm(torch.zeros(1, 4))
    assert torch.isfinite(result).all()


# ---------------------------------------------------------------------------
# npu::causal_conv1d (plain PyTorch, in-place conv-state advance)
# ---------------------------------------------------------------------------


def _scalar_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    bias: torch.Tensor | None,
    width: int,
    length: int,
):
    """Per-sequence reference: (dim, width-1) state + (length, dim) tokens.

    Returns (output (length, dim), new_state (dim, width-1)) in fp32.
    """
    full = torch.cat([state, x.transpose(0, 1)], dim=1)
    out = torch.zeros(full.shape[0], length)
    for position in range(width):
        out = out + full[:, position : position + length] * weight[position].reshape(
            -1, 1
        )
    if bias is not None:
        out = out + bias.reshape(-1, 1)
    return out.transpose(0, 1), full[:, -(width - 1) :]


def test_causal_conv1d_prefill_varlen_matches_scalar_reference() -> None:
    torch.manual_seed(11)
    dim, width = 16, 4
    lengths = [3, 1, 6, 2]
    total = sum(lengths)
    x = (torch.randn(total, dim) * 2).to(torch.bfloat16)
    weight = (torch.randn(width, dim) * 0.5).to(torch.bfloat16)
    bias = (torch.randn(dim) * 0.3).to(torch.bfloat16)
    conv_states = (torch.randn(len(lengths), dim, width - 1) * 0.5).to(torch.bfloat16)
    has_initial_state = torch.tensor([True, True, False, True])
    query_start_loc = torch.tensor([0, *accumulate(lengths)], dtype=torch.long)
    cache_indices = torch.arange(len(lengths))
    state_backup = conv_states.clone()

    out = _fallbacks._causal_conv1d(
        x,
        weight,
        conv_states,
        bias,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation_mode=1,
        pad_slot_id=-1,
        run_mode=0,
    )

    start = 0
    for index, length in enumerate(lengths):
        end = start + length
        state = (
            state_backup[index]
            if bool(has_initial_state[index])
            else torch.zeros(dim, width - 1).to(torch.bfloat16)
        )
        ref_out, ref_state = _scalar_reference(
            x[start:end].float(),
            weight.float(),
            state.float(),
            bias.float(),
            width,
            length,
        )
        assert torch.allclose(
            out[start:end].float(), torch.nn.functional.silu(ref_out), atol=2e-2
        ), f"prefill output mismatch at sequence {index}"
        assert torch.allclose(
            conv_states[index].float(), ref_state.to(torch.bfloat16).float(), atol=2e-2
        ), f"conv-state advance mismatch at sequence {index}"
        start = end


def test_causal_conv1d_decode_vectorizes_and_skips_pad_slots() -> None:
    torch.manual_seed(12)
    dim, width, batch = 8, 3, 5
    x = torch.randn(batch, dim).to(torch.bfloat16)
    weight = (torch.randn(width, dim) * 0.5).to(torch.bfloat16)
    conv_states = torch.randn(batch, dim, width - 1).to(torch.bfloat16)
    query_start_loc = torch.arange(0, batch + 1)
    cache_indices = torch.tensor([0, 1, -1, 3, 4])
    state_backup = conv_states.clone()

    out = _fallbacks._causal_conv1d(
        x,
        weight,
        conv_states,
        None,
        query_start_loc,
        cache_indices,
        None,
        activation_mode=1,
        pad_slot_id=-1,
        run_mode=1,
    )

    for index in range(batch):
        if cache_indices[index] == -1:
            continue  # pad rows: output is undefined by contract, downstream masks
        full = torch.cat(
            [state_backup[index].float(), x[index].float().reshape(dim, 1)], dim=1
        )
        expected = torch.nn.functional.silu((full * weight.float().T).sum(1))
        assert torch.allclose(out[index].float(), expected, atol=2e-2), (
            f"decode output mismatch at row {index}"
        )
        new_state = torch.cat(
            [state_backup[index][:, 1:], x[index].reshape(dim, 1).to(torch.bfloat16)],
            dim=1,
        )
        assert torch.equal(conv_states[cache_indices[index]], new_state), (
            f"decode conv-state mismatch at row {index}"
        )


def test_causal_conv1d_rejects_unsupported_forms() -> None:
    x = torch.zeros(4, 8)
    weight = torch.zeros(3, 8)
    conv_states = torch.zeros(1, 8, 2)
    qsl = torch.tensor([0, 4])
    with pytest.raises(RuntimeError, match="num_accepted_tokens"):
        _fallbacks._causal_conv1d(
            x,
            weight,
            conv_states,
            None,
            qsl,
            None,
            None,
            num_accepted_tokens=torch.tensor([1]),
            activation_mode=0,
        )
    with pytest.raises(RuntimeError, match="activation_mode"):
        _fallbacks._causal_conv1d(
            x, weight, conv_states, None, qsl, None, None, activation_mode=2
        )


# ---------------------------------------------------------------------------
# MTP state-move fallback (CPU): the tile examples/ascend_compat_probe.py
# exercises on the device
# ---------------------------------------------------------------------------


def test_move_intermediate_cache_matches_the_probe_tile() -> None:
    """The probe's exact tile: one request, destination slot 1 <- source step 1."""
    destination = torch.full((1, 2, 2, 128, 128), -1.0)
    source = torch.zeros(1, 1, 2, 2, 128, 128)
    selected = (
        torch.arange(2 * 128 * 128, dtype=torch.float32).reshape(2, 128, 128) / 1024.0
    )
    source[0, 0, 1].copy_(selected)

    moved = _fallbacks.move_intermediate_cache(
        destination,
        source,
        torch.tensor([1]),
        torch.tensor([0]),
        torch.tensor([1]),
        h_block_size=2,
    )

    assert moved is destination
    assert torch.equal(destination[0, 1], selected)
    assert torch.equal(destination[0, 0], torch.full_like(destination[0, 0], -1.0))


def test_move_intermediate_cache_skips_negative_last_steps() -> None:
    """``last_steps < 0`` marks a program as skip: that slot stays untouched."""
    destination = torch.full((2, 3, 4, 3, 3), -7.0)
    source = torch.zeros(2, 2, 2, 4, 3, 3)
    source[:, :, 0] = 1.0
    source[:, :, 1] = 2.0

    _fallbacks.move_intermediate_cache(
        destination,
        source,
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 1, 0]),
        torch.tensor([1, -1, 0]),
    )

    assert torch.equal(destination[:, 0], torch.full_like(destination[:, 0], 2.0))
    assert torch.equal(destination[:, 1], torch.full_like(destination[:, 1], -7.0))
    assert torch.equal(destination[:, 2], torch.full_like(destination[:, 2], 1.0))


def test_move_intermediate_cache_follows_the_source_layout() -> None:
    """The copy resolves through the tensors' own strides, like the vendor kernel."""
    wide = torch.zeros(2, 2, 3, 4, 5, 5)
    source = wide.transpose(1, 2)  # (L, requests, steps, H, V, K), non-contiguous
    source[:, 1, 0] = 3.0
    destination = torch.full((2, 2, 4, 5, 5), -1.0)

    _fallbacks.move_intermediate_cache(
        destination,
        source,
        torch.tensor([1]),
        torch.tensor([1]),
        torch.tensor([0]),
    )

    assert torch.equal(destination[:, 1], torch.full_like(destination[:, 1], 3.0))
    assert torch.equal(destination[:, 0], torch.full_like(destination[:, 0], -1.0))


def test_move_intermediate_cache_rejects_mismatched_indices() -> None:
    """The vendor wrapper asserts equal index lengths; the fallback fails loudly."""
    destination = torch.zeros(1, 2, 2, 4, 4)
    source = torch.zeros(1, 2, 2, 2, 4, 4)
    with pytest.raises(RuntimeError, match="Destination indices lengths must match"):
        _fallbacks.move_intermediate_cache(
            destination,
            source,
            torch.tensor([0, 1]),
            torch.tensor([0]),
            torch.tensor([0]),
        )
    with pytest.raises(RuntimeError, match="Source indices lengths must match"):
        _fallbacks.move_intermediate_cache(
            destination,
            source,
            torch.tensor([0]),
            torch.tensor([0, 1]),
            torch.tensor([0]),
        )


_PATCH_PROBE = """
import importlib
import sys
sys.path.insert(0, {repo_root!r})
import torch
from sglang_fl.dispatch.backends.vendor.ascend import import_guard

assert import_guard.install_import_guards() == ["sgl_kernel_npu"]
from sglang_fl.dispatch.backends.vendor.ascend.patches import mamba_state_update as patch

module = importlib.import_module("sgl_kernel_npu.mamba.mamba_state_update_triton")
assert module.__sglang_fl_placeholder__ is True
assert patch.patch_mamba_state_update_multibuffer() is True
kernel = module.move_cache_dynamic_last_kernel_h_block
assert getattr(kernel.run, "_sglang_fl_cann85_multibuffer_disabled", False) is True
assert patch.patch_mamba_state_update_multibuffer() is True  # idempotent

destination = torch.full((1, 2, 2, 128, 128), -1.0)
source = torch.zeros(1, 1, 2, 2, 128, 128)
source[0, 0, 1] = 1.0
moved = module.move_intermediate_cache(
    destination, source,
    torch.tensor([1]), torch.tensor([0]), torch.tensor([1]), h_block_size=2)
assert moved is destination
assert torch.equal(destination[0, 1], source[0, 0, 1])
assert torch.equal(destination[0, 0], torch.full_like(destination[0, 0], -1.0))
print("probe-ok")
"""


def test_patched_placeholder_answers_the_compat_probe() -> None:
    """Run in a subprocess: through the guard, the patch lands on the stand-in
    kernel and the patched launch performs the state move."""
    result = subprocess.run(
        [sys.executable, "-c", _PATCH_PROBE.format(repo_root=str(_REPO_ROOT))],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "probe-ok" in result.stdout


_REGISTRATION_PROBE = """
import sys
sys.path.insert(0, {repo_root!r})
import torch
from sglang_fl.dispatch.backends.vendor.ascend import sgl_kernel_npu_fallbacks as f

assert f.register_causal_conv1d() is True, "first registration must happen"
assert f.register_causal_conv1d() is False, "registration must be idempotent"

torch.manual_seed(3)
x = torch.randn(5, 8).to(torch.bfloat16)
weight = (torch.randn(3, 8) * 0.5).to(torch.bfloat16)
conv_states = torch.randn(5, 8, 2).to(torch.bfloat16)
qsl = torch.arange(0, 6)
idx = torch.tensor([0, 1, -1, 3, 4])
direct = f._causal_conv1d(x, weight, conv_states, None, qsl, idx, None,
                          activation_mode=1, pad_slot_id=-1, run_mode=1)
conv_states.copy_(torch.randn(5, 8, 2).to(torch.bfloat16).fill_(0) + conv_states)
print("probe-ok")
"""


def test_op_registration_is_idempotent_and_process_persistent() -> None:
    """Run in a subprocess: the registration mutates process-wide torch state."""
    result = subprocess.run(
        [sys.executable, "-c", _REGISTRATION_PROBE.format(repo_root=str(_REPO_ROOT))],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "probe-ok" in result.stdout


# ---------------------------------------------------------------------------
# Triton-delegated GDN fallbacks (device only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _npu_available(), reason="requires an NPU")
def test_fused_gdn_gating_npu_matches_the_closed_form() -> None:
    """g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b), fp32 out."""
    torch.manual_seed(13)
    heads = 8
    a_log = torch.randn(heads, device="npu", dtype=torch.float32)
    a = torch.randn(7, heads, device="npu", dtype=torch.bfloat16)
    b = torch.randn(7, heads, device="npu", dtype=torch.bfloat16)
    dt_bias = torch.randn(heads, device="npu", dtype=torch.float32)

    g, beta = _fallbacks.fused_gdn_gating_npu(a_log, a, b, dt_bias)

    x = a.float() + dt_bias.float()
    softplus = torch.where(x <= 20.0, torch.log(1.0 + torch.exp(x)) / 1.0, x)
    expected_g = -torch.exp(a_log.float()) * softplus
    assert g.shape == (1, 7, heads)
    assert torch.allclose(g[0], expected_g, atol=2e-2)
    assert beta.shape == (1, 7, heads)
    assert torch.allclose(beta[0], torch.sigmoid(b.float()), atol=2e-2)


@pytest.mark.skipif(not _npu_available(), reason="requires an NPU")
def test_chunk_gated_delta_rule_npu_delegates_to_the_triton_kernel() -> None:
    """The placeholder name runs SGLang's Triton chunk kernel end to end, in
    the production NPU call form: gathered states, no ``initial_state_indices``
    (the fallback bridges it to the pool form the kernel expects)."""
    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule

    torch.manual_seed(14)
    batch, tokens, heads, key_dim, value_dim = 1, 8, 4, 32, 64
    pool = torch.randn(3, heads, key_dim, value_dim, device="npu", dtype=torch.float32)
    gathered = pool[[1]]  # the gdn_triton.py NPU form: ssm_states[cache_indices]
    q = torch.randn(batch, tokens, heads, key_dim, device="npu", dtype=torch.bfloat16)
    k = torch.randn(batch, tokens, heads, key_dim, device="npu", dtype=torch.bfloat16)
    v = torch.randn(batch, tokens, heads, value_dim, device="npu", dtype=torch.bfloat16)
    g = torch.randn(batch, tokens, heads, device="npu", dtype=torch.float32)
    beta = torch.rand(batch, tokens, heads, device="npu", dtype=torch.bfloat16)

    output, _, final_state = _fallbacks.chunk_gated_delta_rule_npu(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=key_dim**-0.5,
        initial_state=gathered,
    )
    reference, _, reference_state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=key_dim**-0.5,
        initial_state=pool,
        initial_state_indices=torch.tensor([1], device="npu"),
    )
    assert output.shape == reference.shape
    assert torch.allclose(output.float(), reference.float(), atol=2e-2)
    torch.testing.assert_close(
        final_state.float(), reference_state.float(), rtol=2e-2, atol=2e-2
    )
