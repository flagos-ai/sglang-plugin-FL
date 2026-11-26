# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TXDA dummy-weight patch: CPU-generated values, upstream contract (no device work)."""

import pytest
import torch

loader = pytest.importorskip("sglang.srt.model_loader.loader")
weight_utils = pytest.importorskip("sglang.srt.model_loader.weight_utils")

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches import (  # noqa: E402
    dummy_weights as dw,
)


class _FakeTxdaTensor:
    """CPU tensor that reports device type 'txda' to the patch."""

    def __init__(self, tensor):
        self.data = tensor
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.device = type("D", (), {"type": "txda"})()


class _Model(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self._state = state

    def state_dict(self):
        return self._state


def _txda_param(shape, dtype):
    return _FakeTxdaTensor(torch.zeros(shape, dtype=dtype))


@pytest.fixture(autouse=True)
def _floating(monkeypatch):
    # torch.is_floating_point needs a real tensor; route the fake to its data.
    real = torch.is_floating_point
    monkeypatch.setattr(
        torch,
        "is_floating_point",
        lambda t: real(t.data if isinstance(t, _FakeTxdaTensor) else t),
    )


def test_txda_params_get_cpu_seeded_values() -> None:
    fn = dw._make_txda_aware(weight_utils.initialize_dummy_weights)
    a, b = _txda_param((4, 8), torch.bfloat16), _txda_param((4, 8), torch.bfloat16)
    fn(_Model({"a": a, "b": b}), low=-1e-3, high=1e-3, seed=7)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(7)
    expected = torch.empty((4, 8), dtype=torch.bfloat16).uniform_(-1e-3, 1e-3, generator=gen)
    # Per-parameter seed: same shape/dtype -> same values (TP-rank consistency).
    assert torch.equal(a.data, expected)
    assert torch.equal(b.data, expected)
    assert a.data.abs().max() <= 1e-3 and a.data.abs().sum() > 0


def test_low_bit_params_use_fp16_generation() -> None:
    fn = dw._make_txda_aware(weight_utils.initialize_dummy_weights)
    p = _txda_param((16,), torch.float8_e4m3fn)
    # +-1e-3 underflows to 0 in fp8 e4m3 (upstream too); use a visible range.
    fn(_Model({"p": p}), low=-1.0, high=1.0, seed=3)
    assert p.data.dtype == torch.float8_e4m3fn
    assert p.data.to(torch.float32).abs().sum() > 0


def test_non_float_and_non_txda_params_use_upstream(monkeypatch) -> None:
    calls = []

    def fake_upstream(model, low, high, seed):
        calls.append(sorted(model.state_dict()))

    fn = dw._make_txda_aware(fake_upstream)
    ints = _txda_param((3,), torch.int32)
    fn(_Model({"cpu_w": torch.zeros(2), "idx": ints, "t": _txda_param((2,), torch.float16)}))
    assert calls == [["cpu_w"]]
    assert torch.equal(ints.data, torch.zeros(3, dtype=torch.int32))


def test_apply_rebinds_loader_and_weight_utils(monkeypatch) -> None:
    orig = weight_utils.initialize_dummy_weights
    monkeypatch.setattr(loader, "initialize_dummy_weights", orig)
    monkeypatch.setattr(weight_utils, "initialize_dummy_weights", orig)
    dw._apply()
    assert loader.initialize_dummy_weights.__wrapped__ is orig
    assert weight_utils.initialize_dummy_weights is loader.initialize_dummy_weights
    patched = loader.initialize_dummy_weights
    dw._apply()  # idempotent
    assert loader.initialize_dummy_weights is patched
