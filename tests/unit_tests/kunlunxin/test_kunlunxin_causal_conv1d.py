# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for ``patches/causal_conv1d.py``.

The GDN causal conv is the layer whose layout/dtype bridging broke most often on
Kunlunxin (NCW vs NWC, fp32-only klx update kernel, ``conv_state_indices``
handling). The XPU kernels are replaced by reference depthwise convolutions that
consume exactly the layout the real kernel consumes, so each test pins:

  * the NCW (sglang) <-> NWC (kernel) transposes for ``x`` and ``conv_states``,
  * the int32 device/CPU ``cu_seqlens`` pair and int32 ``cache_indices``,
  * the fp32 shadow used by the decode update kernel,
  * in-place write-back of the updated conv state.
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.mamba.causal_conv1d_triton import PAD_SLOT_ID

from sglang_fl.dispatch.backends.vendor.kunlunxin.patches import causal_conv1d


def _depthwise_causal_conv(full, weight, bias, silu):
    """``full``: (N, dim) NWC rows; ``weight``: (dim, W) -> (N-W+1, dim)."""
    window = full.unfold(0, weight.shape[1], 1)
    out = (window * weight.unsqueeze(0)).sum(-1)
    if bias is not None:
        out = out + bias
    if silu:
        out = F.silu(out)
    return out


@pytest.fixture
def fla_xpu_ops(monkeypatch):
    """Reference implementation of ``fla_xpu_ops.causal_conv1d_fwd``."""
    record = {}
    module = types.ModuleType("fla_xpu_ops")

    def causal_conv1d_fwd(
        x,
        weight,
        cu_seqlens,
        cu_seqlens_cpu,
        cache_indices,
        has_initial_state,
        silu_activation,
        is_ncw,
        pad_slot_id,
        bias,
        conv_states,
        dilation,
        out,
    ):
        record.update(
            x=x,
            weight=weight,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            silu=silu_activation,
            is_ncw=is_ncw,
            pad_slot_id=pad_slot_id,
            bias=bias,
            conv_states=conv_states,
            dilation=dilation,
            out=out,
        )
        assert is_ncw is False, "kunlunxin bridge must feed NWC"
        assert conv_states.shape[-1] == x.shape[-1]

        bounds = cu_seqlens_cpu.tolist()
        states = conv_states.clone()
        pieces = []
        for index in range(len(bounds) - 1):
            start, end = bounds[index], bounds[index + 1]
            cache = int(cache_indices[index])
            if int(has_initial_state[index]):
                prefix = states[cache]
            else:
                prefix = torch.zeros_like(states[cache])
            full = torch.cat([prefix, x[start:end]], 0)
            pieces.append(
                _depthwise_causal_conv(full, weight, bias, silu_activation)[
                    -(end - start) :
                ]
            )
            if states.shape[1]:
                states[cache] = full[-states.shape[1] :]
        out.copy_(torch.cat(pieces, 0).to(out.dtype))
        conv_states.copy_(states)

    module.causal_conv1d_fwd = causal_conv1d_fwd
    monkeypatch.setitem(sys.modules, "fla_xpu_ops", module)
    return record


@pytest.fixture
def klx_update(monkeypatch):
    """Reference implementation of ``torch.ops.sgl_kernel.causal_conv1d_update``."""
    record = {}
    fake_update = SimpleNamespace()

    def update(
        x,
        conv_states,
        weight,
        bias,
        silu_activation,
        cache_seqlens,
        conv_state_indices,
        intermediate_conv_window,
        is_ncw,
        pad_slot_id,
    ):
        record.update(
            x=x,
            conv_states=conv_states,
            weight=weight,
            bias=bias,
            silu=silu_activation,
            cache_seqlens=cache_seqlens,
            conv_state_indices=conv_state_indices,
            intermediate_conv_window=intermediate_conv_window,
            is_ncw=is_ncw,
            pad_slot_id=pad_slot_id,
        )
        assert is_ncw is False, "klx update kernel only accepts NWC"
        assert x.dtype == torch.float32, "decode bridge must use the fp32 shadow"

        states = conv_states.clone()
        rows = []
        for row in range(x.shape[0]):
            cache = int(conv_state_indices[row])
            full = torch.cat([states[cache], x[row]], 0)
            rows.append(_depthwise_causal_conv(full, weight, bias, silu_activation)[-1])
            states[cache] = full[-conv_states.shape[1] :]
        x.copy_(torch.stack(rows, 0).unsqueeze(1))
        conv_states.copy_(states)

    fake_update.default = update
    monkeypatch.setattr(
        torch.ops, "sgl_kernel", SimpleNamespace(causal_conv1d_update=fake_update),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "sgl_kernel", types.ModuleType("sgl_kernel"))
    return record


def _prefill_inputs(seq_lens, dim=8, width=4, state_len=4, dtype=torch.float16, lines=2):
    torch.manual_seed(0)
    total = sum(seq_lens)
    x = torch.randn(dim, total, dtype=dtype)  # NCW, as gdn_backend passes it
    weight = torch.randn(dim, width, dtype=dtype)
    bias = torch.randn(dim, dtype=dtype)
    conv_states = torch.randn(lines, dim, state_len, dtype=dtype)
    cu_seqlens = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int64)
    return SimpleNamespace(
        x=x,
        weight=weight,
        bias=bias,
        conv_states=conv_states,
        cu_seqlens=cu_seqlens,
        cache_indices=torch.arange(len(seq_lens), dtype=torch.int32),
        has_initial_state=torch.zeros(len(seq_lens), dtype=torch.bool),
    )


def test_prefill_bridges_ncw_to_nwc_and_writes_back_state(fla_xpu_ops):
    inputs = _prefill_inputs([3, 4])
    state_len = inputs.conv_states.shape[2]
    expected_states_nwc = inputs.conv_states.transpose(1, 2).clone()
    expected_out = []
    bounds = inputs.cu_seqlens.tolist()
    for index in range(len(bounds) - 1):
        start, end = bounds[index], bounds[index + 1]
        cache = int(inputs.cache_indices[index])
        full = torch.cat(
            [torch.zeros_like(expected_states_nwc[cache]), inputs.x[:, start:end].t()], 0
        )
        expected_out.append(
            _depthwise_causal_conv(full, inputs.weight, inputs.bias, True)[-(end - start) :]
        )
        expected_states_nwc[cache] = full[-state_len:]
    expected_out = torch.cat(expected_out, 0)

    out = causal_conv1d._kunlun_causal_conv1d_fn(
        inputs.x,
        inputs.weight,
        inputs.bias,
        query_start_loc=inputs.cu_seqlens,
        cache_indices=inputs.cache_indices,
        has_initial_state=inputs.has_initial_state,
        conv_states=inputs.conv_states,
        activation="silu",
    )

    assert out.shape == inputs.x.shape  # back to NCW for the caller
    assert torch.allclose(out.t().float(), expected_out.float(), atol=1e-3)
    # State cache is NCW again and updated in place.
    assert torch.allclose(
        inputs.conv_states.transpose(1, 2).float(),
        expected_states_nwc.float(),
        atol=1e-3,
    )
    assert fla_xpu_ops["dilation"] == 1
    assert fla_xpu_ops["pad_slot_id"] == PAD_SLOT_ID
    assert fla_xpu_ops["silu"] is True
    assert fla_xpu_ops["cu_seqlens"].dtype == torch.int32
    assert fla_xpu_ops["cu_seqlens_cpu"].dtype == torch.int32
    assert fla_xpu_ops["cu_seqlens_cpu"].device.type == "cpu"
    assert fla_xpu_ops["cache_indices"].dtype == torch.int32
    assert fla_xpu_ops["has_initial_state"].dtype == torch.int32
    assert fla_xpu_ops["weight"].dtype == inputs.x.dtype
    assert fla_xpu_ops["conv_states"].shape == (2, 4, 8)  # NWC view


def test_prefill_seeds_from_existing_state_when_requested(fla_xpu_ops):
    inputs = _prefill_inputs([3])
    inputs.has_initial_state = torch.ones(1, dtype=torch.bool)
    prefix = inputs.conv_states[0].clone()  # the call rewrites this cache line

    out = causal_conv1d._kunlun_causal_conv1d_fn(
        inputs.x,
        inputs.weight,
        inputs.bias,
        query_start_loc=inputs.cu_seqlens,
        cache_indices=inputs.cache_indices,
        has_initial_state=inputs.has_initial_state,
        conv_states=inputs.conv_states,
        activation=None,
    )

    full = torch.cat([prefix.transpose(0, 1), inputs.x.t()], 0)
    expected = _depthwise_causal_conv(full, inputs.weight, inputs.bias, False)[-3:]
    assert torch.allclose(out.t().float(), expected.float(), atol=1e-3)


def test_prefill_derives_start_loc_from_seq_lens_cpu(fla_xpu_ops):
    inputs = _prefill_inputs([2, 2])

    causal_conv1d._kunlun_causal_conv1d_fn(
        inputs.x,
        inputs.weight,
        inputs.bias,
        seq_lens_cpu=[2, 2],
        cache_indices=inputs.cache_indices,
        conv_states=inputs.conv_states,
        activation="silu",
    )

    assert fla_xpu_ops["cu_seqlens"].tolist() == [0, 2, 4]
    assert fla_xpu_ops["has_initial_state"].tolist() == [0, 0]


@pytest.mark.parametrize("activation", ["gelu", "relu"])
def test_prefill_rejects_unsupported_activation(fla_xpu_ops, activation):
    inputs = _prefill_inputs([2])
    with pytest.raises(NotImplementedError, match="activation must be"):
        causal_conv1d._kunlun_causal_conv1d_fn(
            inputs.x,
            inputs.weight,
            inputs.bias,
            conv_states=inputs.conv_states,
            activation=activation,
        )


def test_decode_runs_in_fp32_nwc_and_returns_model_dtype(klx_update):
    torch.manual_seed(1)
    batch, dim, width, state_len = 2, 8, 4, 4
    x = torch.randn(batch, dim, dtype=torch.float16)  # 2D, as gdn_backend passes it
    weight = torch.randn(dim, width, dtype=torch.float16)
    bias = torch.randn(dim, dtype=torch.float16)
    conv_states = torch.randn(3, dim, state_len, dtype=torch.float16)
    indices = torch.tensor([2, 0], dtype=torch.int32)
    expected_states = conv_states.clone()
    untouched = conv_states[1].clone()

    out = causal_conv1d._kunlun_causal_conv1d_update(
        x,
        conv_states,
        weight,
        bias,
        "silu",
        None,
        indices,
    )

    assert out.dtype == torch.float16
    assert out.shape == x.shape
    for row in range(batch):
        cache = int(indices[row])
        full = torch.cat([expected_states[cache].t().float(), x[row].float().unsqueeze(0)], 0)
        expected = _depthwise_causal_conv(full, weight.float(), bias.float(), True)[-1]
        assert torch.allclose(out[row].float(), expected, atol=2e-2)
        expected_states[cache] = full[-state_len:].t().to(torch.float16)

    assert torch.allclose(conv_states, expected_states, atol=2e-2)
    # Untouched cache line stays as it was.
    assert torch.equal(conv_states[1], untouched)
    assert klx_update["x"].shape == (batch, 1, dim)  # seqlen axis added
    assert klx_update["conv_states"].shape == (3, state_len, dim)  # NWC
    assert klx_update["pad_slot_id"] == PAD_SLOT_ID
    assert klx_update["intermediate_conv_window"] is None


def test_decode_without_activation_skips_silu(klx_update):
    torch.manual_seed(2)
    x = torch.randn(1, 8, dtype=torch.float32)
    snapshot = x.clone()  # an fp32 input is consumed in place by the kernel
    weight = torch.randn(8, 4)
    conv_states = torch.zeros(1, 8, 4)

    out = causal_conv1d._kunlun_causal_conv1d_update(
        x, conv_states, weight, None, None, None, torch.zeros(1, dtype=torch.int32)
    )

    full = torch.cat([torch.zeros(4, 8), snapshot], 0)
    expected = _depthwise_causal_conv(full, weight, None, False)[-1]
    assert torch.allclose(out[0], expected, atol=1e-4)
    assert klx_update["silu"] is False
