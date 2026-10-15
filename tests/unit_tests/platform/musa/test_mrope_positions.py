# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import torch

from sglang_fl.dispatch.backends.vendor.mthreads.patches import mrope_positions


class _ForwardMode:
    def __init__(self, is_decode: bool):
        self._is_decode = is_decode

    def is_decode(self):
        return self._is_decode


def _make_forward_batch(batch_size: int, *, is_decode: bool = True):
    return SimpleNamespace(
        forward_mode=_ForwardMode(is_decode),
        positions=torch.arange(batch_size, dtype=torch.int64),
        mrope_positions=None,
    )


def test_text_decode_reuses_device_positions(monkeypatch):
    monkeypatch.delenv(
        "SGLANG_MUSA_DECODE_MROPE_DEVICE_POSITIONS", raising=False
    )
    forward_batch = _make_forward_batch(64)
    worker_batch = SimpleNamespace(multimodal_inputs=[None] * 64)
    calls = []

    def original(*args):
        calls.append(args)

    wrapped = mrope_positions._wrap_compute_mrope_positions(
        original,
        lambda: SimpleNamespace(rl_on_policy_target=None),
    )
    wrapped(forward_batch, object(), worker_batch)

    expected = forward_batch.positions.unsqueeze(0).repeat(3, 1)
    torch.testing.assert_close(forward_batch.mrope_positions, expected, rtol=0, atol=0)
    assert forward_batch.mrope_positions.shape == (3, 64)
    assert forward_batch.mrope_positions.stride() == (0, 1)
    assert calls == []


def test_multimodal_and_prefill_delegate():
    sentinel = object()

    def original(*args):
        return sentinel

    wrapped = mrope_positions._wrap_compute_mrope_positions(
        original,
        lambda: SimpleNamespace(rl_on_policy_target=None),
    )

    multimodal_decode = _make_forward_batch(1)
    worker_batch = SimpleNamespace(multimodal_inputs=[object()])
    assert wrapped(multimodal_decode, object(), worker_batch) is sentinel

    text_prefill = _make_forward_batch(1, is_decode=False)
    worker_batch = SimpleNamespace(multimodal_inputs=[None])
    assert wrapped(text_prefill, object(), worker_batch) is sentinel


def test_device_positions_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SGLANG_MUSA_DECODE_MROPE_DEVICE_POSITIONS", "off")
    sentinel = object()

    def original(*args):
        return sentinel

    wrapped = mrope_positions._wrap_compute_mrope_positions(
        original,
        lambda: SimpleNamespace(rl_on_policy_target=None),
    )
    forward_batch = _make_forward_batch(8)
    worker_batch = SimpleNamespace(multimodal_inputs=[None] * 8)

    assert wrapped(forward_batch, object(), worker_batch) is sentinel
    assert forward_batch.mrope_positions is None
