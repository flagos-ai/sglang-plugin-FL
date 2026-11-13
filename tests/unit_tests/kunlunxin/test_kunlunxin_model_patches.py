# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the two model/scheduler-level Kunlunxin patches.

  * ``patches/pp_send_first.py``     — PP ring ordering must not deadlock when
    ``isend`` is effectively blocking; Kunlunxin keys it off
    ``pp_group.is_first_rank`` instead of ``pp_rank`` parity.
  * ``patches/qwen35_attn_native.py`` — Qwen3.5/3.6 full-attention prepare must
    avoid the CUDA-fused Triton QK-norm/RoPE/gate kernel on cuda-compat Kunlun.
"""

import contextlib
from types import SimpleNamespace

import torch

from sglang_fl.dispatch.backends.vendor.kunlunxin.patches import (
    pp_send_first,
    qwen35_attn_native,
)


# ─── pp_send_first ───────────────────────────────────────────────────────────


class _Stream:
    def __init__(self):
        self.waited = None

    def wait_stream(self, other):
        self.waited = other


class _Event:
    def __init__(self):
        self.recorded = None

    def record(self, stream):
        self.recorded = stream


class _DeviceModule:
    def __init__(self):
        self.stream = "schedule-stream"

    def Event(self):
        return _Event()

    def current_stream(self):
        return self.stream


def _make_scheduler(is_first_rank, order, *, prebuilt=False):
    scheduler = SimpleNamespace()
    scheduler.pp_group = SimpleNamespace(is_first_rank=is_first_rank)
    scheduler.copy_stream_ctx = contextlib.nullcontext()
    scheduler.copy_stream = _Stream()
    scheduler.schedule_stream = "schedule-stream"
    scheduler.device_module = _DeviceModule()

    def send(*args):
        order.append("send")
        return ["send-work"]

    def recv(*args):
        order.append("recv")
        return {"hidden_states": torch.zeros(1)}

    def prep(batch, metadata, outputs):
        order.append("prep")
        return "batch-result"

    scheduler._pp_send_output_to_next_stage = send
    scheduler._pp_recv_dict_from_prev_stage = recv
    scheduler._pp_prep_batch_result = prep

    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_prebuilt=lambda: prebuilt)
    )
    return scheduler, [batch], [None]


def _call(scheduler, mbs, metadata, order):
    return pp_send_first._pp_send_recv_and_preprocess_output_tensors(
        scheduler,
        next_first_rank_mb_id=0,
        next_mb_id=0,
        mbs=mbs,
        mb_metadata=metadata,
        last_rank_comm_queue=[],
        pp_outputs=["pp-output"],
    )


def test_first_pp_rank_sends_before_receiving():
    order = []
    scheduler, mbs, metadata = _make_scheduler(True, order)

    next_outputs, batch_result, event, send_work = _call(
        scheduler, mbs, metadata, order
    )

    assert order == ["send", "recv", "prep"]
    assert send_work == ["send-work"]
    assert batch_result == "batch-result"
    assert next_outputs is not None
    assert type(next_outputs).__name__ == "PPProxyTensors"
    assert event.recorded == "schedule-stream"


def test_later_pp_ranks_receive_before_sending():
    order = []
    scheduler, mbs, metadata = _make_scheduler(False, order)

    next_outputs, batch_result, _, send_work = _call(scheduler, mbs, metadata, order)

    assert order == ["recv", "prep", "send"]
    assert send_work == ["send-work"]
    assert batch_result == "batch-result"
    assert next_outputs is not None


def test_prebuilt_microbatch_skips_the_receive():
    order = []
    scheduler, mbs, metadata = _make_scheduler(False, order, prebuilt=True)

    next_outputs, batch_result, event, send_work = _call(
        scheduler, mbs, metadata, order
    )

    assert order == ["send"]
    assert next_outputs is None
    assert batch_result is None
    assert event is None
    assert send_work == ["send-work"]


def test_patch_rebinds_the_scheduler_mixin(monkeypatch):
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    monkeypatch.setattr(
        SchedulerPPMixin,
        "_pp_send_recv_and_preprocess_output_tensors",
        SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors,
    )
    monkeypatch.setattr(pp_send_first, "_applied", False)

    pp_send_first.patch_pp_send_recv_and_preprocess_output_tensors()

    assert (
        SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors
        is pp_send_first._pp_send_recv_and_preprocess_output_tensors
    )


# ─── qwen35_attn_native ──────────────────────────────────────────────────────


def _make_attention_layer(*, gate, attn_output, o_proj_output, calls):
    layer = SimpleNamespace()
    layer.attn_output_gate = gate is not None

    def forward_prepare_native(positions, hidden_states):
        calls.append(("prepare", positions, hidden_states))
        return (
            torch.zeros_like(hidden_states),
            torch.zeros_like(hidden_states),
            torch.zeros_like(hidden_states),
            gate,
        )

    def attn(q, k, v, forward_batch):
        calls.append(("attn", forward_batch))
        return attn_output

    def o_proj(x):
        calls.append(("o_proj", x))
        return o_proj_output, None

    layer.forward_prepare_native = forward_prepare_native
    layer.attn = attn
    layer.o_proj = o_proj
    return layer


def test_native_prepare_uses_torch_sigmoid_gate_for_3d_gate():
    calls = []
    attn_output = torch.randn(3, 8)
    gate = torch.randn(3, 4, 2)  # strided 3-D gate, as forward_prepare_native returns
    o_proj_output = torch.randn(3, 8)
    layer = _make_attention_layer(
        gate=gate, attn_output=attn_output, o_proj_output=o_proj_output, calls=calls
    )
    positions = torch.arange(3)
    hidden_states = torch.randn(3, 16)
    forward_batch = SimpleNamespace(batch_size=1)

    result = qwen35_attn_native._native_self_attention(
        layer, positions, hidden_states, forward_batch
    )

    expected = attn_output * torch.sigmoid(gate.reshape(3, -1))
    o_proj_input = [entry[1] for entry in calls if entry[0] == "o_proj"][0]
    assert torch.allclose(o_proj_input, expected)
    assert torch.equal(result, o_proj_output)
    assert [entry[0] for entry in calls] == ["prepare", "attn", "o_proj"]


def test_native_prepare_without_gate_passes_attention_output_through():
    calls = []
    attn_output = torch.randn(2, 8)
    layer = _make_attention_layer(
        gate=None, attn_output=attn_output, o_proj_output=torch.randn(2, 8), calls=calls
    )

    qwen35_attn_native._native_self_attention(
        layer, torch.arange(2), torch.randn(2, 16), SimpleNamespace()
    )

    o_proj_input = [entry[1] for entry in calls if entry[0] == "o_proj"][0]
    assert o_proj_input is attn_output


def test_native_prepare_passes_positions_and_hidden_states_by_keyword():
    calls = []
    layer = _make_attention_layer(
        gate=None,
        attn_output=torch.randn(1, 4),
        o_proj_output=torch.randn(1, 4),
        calls=calls,
    )
    positions = torch.arange(1)
    hidden_states = torch.randn(1, 8)

    qwen35_attn_native._native_self_attention(
        layer, positions, hidden_states, SimpleNamespace()
    )

    assert calls[0] == ("prepare", positions, hidden_states)


def test_patch_rebinds_qwen35_full_attention_layer(monkeypatch):
    from sglang.srt.models.qwen3_5 import Qwen3_5AttentionDecoderLayer

    monkeypatch.setattr(
        Qwen3_5AttentionDecoderLayer,
        "self_attention",
        Qwen3_5AttentionDecoderLayer.self_attention,
    )
    monkeypatch.setattr(qwen35_attn_native, "_applied", False)

    qwen35_attn_native.patch_qwen35_attn_native()

    assert (
        Qwen3_5AttentionDecoderLayer.self_attention
        is qwen35_attn_native._native_self_attention
    )
