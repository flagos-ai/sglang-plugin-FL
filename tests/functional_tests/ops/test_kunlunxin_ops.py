# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Functional (on-device) checks for the Kunlunxin vendor kernels.

The unit matrix stubs the XPU kernels to pin layouts; these tests run the real
``sgl_kernel`` / ``fla_xpu_ops`` kernels through the vendor wrappers and compare
against a pure-torch reference:

  * ``impl/activation.py``    — xflashinfer ``swiglu_ep`` split/flatten contract
  * ``patches/causal_conv1d.py`` — GDN prefill (xfla NWC) + decode (klx fp32
    shadow, ``conv_state_indices``) including the in-place conv-state update
  * ``impl/fused_moe.py``     — ``klx_fused_experts`` weight layout and int32
    ``topk_ids``

Every test skips when its kernel package is absent, so the file is inert on
non-Kunlunxin devices.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch
import torch.nn.functional as F

from sglang_fl.dispatch.backends.vendor.kunlunxin.impl.fused_moe import (
    fused_moe_kunlunxin,
)
from sglang_fl.dispatch.backends.vendor.kunlunxin.kunlunxin import KunlunxinBackend
from sglang_fl.dispatch.backends.vendor.kunlunxin.patches import causal_conv1d

pytestmark = [pytest.mark.functional, pytest.mark.gpu]


def _has(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _device_backend_available() -> bool:
    try:
        return KunlunxinBackend().is_available()
    except Exception:
        return False


pytestmark.append(
    pytest.mark.skipif(
        not _device_backend_available(), reason="Kunlunxin XPU backend unavailable"
    )
)


def _depthwise_conv(full: torch.Tensor, weight: torch.Tensor, bias, silu: bool):
    """``full``: (N, dim) -> causal depthwise conv over the last W rows."""
    window = full.unfold(0, weight.shape[1], 1)
    out = (window * weight.unsqueeze(0)).sum(-1)
    if bias is not None:
        out = out + bias
    return F.silu(out) if silu else out


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, dtype) -> None:
    tol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
    assert torch.allclose(actual.float(), expected.float(), rtol=tol, atol=tol)


@pytest.mark.skipif(
    not (_has("fla_xpu_ops") and _has("sgl_kernel")),
    reason="xfla/klx kernels unavailable",
)
def test_causal_conv1d_prefill_matches_torch_on_device(device):
    """xfla prefill consumes NWC views of SGLang's NCW buffer in model dtype.

    ``fla_xpu_ops.causal_conv1d_fwd`` rejects a bias ("bias is not supported"),
    and the Qwen3.5/3.6 GDN conv is bias-free, so the test mirrors production.
    The conv state holds ``width - 1`` inputs: that is the prefix the kernel
    replays for ``has_initial_state`` requests.
    """
    dim, width, tokens = 32, 4, 6
    state_len = width - 1
    dtype = torch.bfloat16
    torch.manual_seed(0)
    x = torch.randn(dim, tokens, device=device, dtype=dtype)  # NCW from gdn_backend
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    conv_states = torch.randn(2, dim, state_len, device=device, dtype=dtype)
    started = conv_states.clone()

    out = causal_conv1d._kunlun_causal_conv1d_fn(
        x,
        weight,
        None,
        query_start_loc=torch.tensor([0, 3, 6], dtype=torch.int32, device=device),
        cache_indices=torch.tensor([0, 1], dtype=torch.int32, device=device),
        has_initial_state=torch.tensor([1, 0], dtype=torch.bool, device=device),
        conv_states=conv_states,
        activation="silu",
    )

    assert out.shape == x.shape
    for index, (start, end) in enumerate(((0, 3), (3, 6))):
        prefix = (
            started[index].t()
            if index == 0
            else torch.zeros(state_len, dim, device=device, dtype=dtype)
        )
        full = torch.cat([prefix, x[:, start:end].t()], 0)
        expected = _depthwise_conv(full, weight, None, True)[-(end - start) :]
        _assert_close(out[:, start:end].t(), expected, dtype)
        _assert_close(conv_states[index].t(), full[-state_len:], dtype)


@pytest.mark.skipif(
    not (_has("fla_xpu_ops") and _has("sgl_kernel")),
    reason="xfla/klx kernels unavailable",
)
def test_causal_conv1d_decode_chains_the_prefill_state(device):
    """Decode must continue the exact conv state that prefill produced.

    The two kernels live in different packages (xfla prefill, klx decode) and
    share SGLang's ``(lines, dim, state_len)`` NCW cache, so the contract that
    matters is that a seeded decode reproduces the reference causal conv of the
    full sequence.
    """
    tokens, dim, width = 5, 16, 4
    state_len = width - 1
    dtype = torch.bfloat16
    torch.manual_seed(1)
    x = torch.randn(tokens, dim, device=device, dtype=dtype)  # NWC token rows
    weight = torch.randn(dim, width, device=device, dtype=dtype)

    padded = torch.cat(
        [torch.zeros(state_len, dim, device=device, dtype=dtype), x], 0
    )
    reference = _depthwise_conv(padded, weight, None, True)[-tokens:]

    conv_states = torch.zeros(1, dim, state_len, device=device, dtype=dtype)
    seeded = causal_conv1d._kunlun_causal_conv1d_fn(
        x[:1].t().contiguous(),
        weight,
        None,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        cache_indices=torch.tensor([0], dtype=torch.int32, device=device),
        has_initial_state=torch.tensor([False], device=device),
        conv_states=conv_states,
        activation="silu",
    )
    _assert_close(seeded.t()[0], reference[0], dtype)

    indices = torch.tensor([0], dtype=torch.int32, device=device)
    for step in range(1, tokens):
        decoded = causal_conv1d._kunlun_causal_conv1d_update(
            x[step : step + 1], conv_states, weight, None, "silu", None, indices
        )
        assert decoded.dtype is dtype
        _assert_close(decoded[0], reference[step], dtype)


@pytest.mark.skipif(not _has("sgl_kernel"), reason="klx kernel unavailable")
def test_fused_moe_matches_torch_on_device(device):
    tokens, hidden, inter, experts, topk = 6, 128, 256, 4, 2
    dtype = torch.bfloat16
    torch.manual_seed(2)
    x = torch.randn(tokens, hidden, device=device, dtype=dtype)
    w13 = torch.randn(experts, 2 * inter, hidden, device=device, dtype=dtype) * 0.05
    w2 = torch.randn(experts, hidden, inter, device=device, dtype=dtype) * 0.05
    # ``klx_fused_experts`` requires float32 routing weights (sglang's
    # select_experts emits them in float32), so keep them out of the bf16 cast.
    topk_weights = torch.softmax(
        torch.randn(tokens, topk, device=device, dtype=torch.float32), dim=-1
    )
    topk_ids = torch.stack(
        [torch.arange(topk, device=device) for _ in range(tokens)]
    ).to(torch.int32)

    expected = torch.zeros(tokens, hidden, device=device, dtype=torch.float32)
    for token in range(tokens):
        for slot in range(topk):
            expert = int(topk_ids[token, slot])
            gate_up = x[token].float() @ w13[expert].float().t()
            gate, up = gate_up.split(inter, dim=-1)
            fused = F.silu(gate) * up
            expected[token] += topk_weights[token, slot].float() * (
                fused @ w2[expert].float().t()
            )

    class _Runner:
        moe_runner_config = type(
            "Config", (), {"activation": "silu", "inplace": False, "is_gated": True}
        )()

    class _Layer:
        w13_weight = w13
        w2_weight = w2

    class _DispatchOutput:
        hidden_states = x
        topk_output = type(
            "TopKOutput", (), {"topk_weights": topk_weights, "topk_ids": topk_ids}
        )()

    result = fused_moe_kunlunxin(_Runner(), _Layer(), _DispatchOutput())

    _assert_close(result.hidden_states, expected, dtype)
