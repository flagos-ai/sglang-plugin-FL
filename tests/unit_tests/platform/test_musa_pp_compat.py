# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""PP parity ordering and skipped-output semantics across scheduler APIs."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from sglang_fl.dispatch.backends.vendor.mthreads.patch import _patch_pp_send_recv_order


@pytest.mark.parametrize("new_api", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_pp_order_and_skipped_output(monkeypatch, new_api, rank):
    module = pytest.importorskip("sglang.srt.managers.scheduler_pp_mixin")
    cls = module.SchedulerPPMixin
    monkeypatch.setattr(
        cls, "_pp_send_recv_and_preprocess_output_tensors", lambda *args: None
    )
    state = {"skip": False}
    if new_api:
        monkeypatch.setattr(
            module,
            "_pp_can_skip_output_comm",
            lambda batch: state["skip"],
            raising=False,
        )
    else:
        monkeypatch.delattr(module, "_pp_can_skip_output_comm", raising=False)
    _patch_pp_send_recv_order()

    calls = []

    def send(*args):
        calls.append("send")
        return ["work"]

    def recv():
        calls.append("recv")
        return {"value": "output"}

    event = SimpleNamespace(record=lambda stream: None)
    scheduler = SimpleNamespace(
        _pp_send_output_to_next_stage=send,
        _pp_recv_dict_from_prev_stage=recv,
        _pp_prep_batch_result=lambda *args: "batch",
        _pp_make_skip_output_result=lambda *args: ("skipped", "batch", event),
        copy_stream_ctx=nullcontext(),
        copy_stream=SimpleNamespace(wait_stream=lambda stream: None),
        schedule_stream=object(),
        device_module=SimpleNamespace(Event=lambda: event, current_stream=lambda: None),
    )
    if new_api:
        scheduler.ps = SimpleNamespace(pp_rank=rank)
    else:
        scheduler.pp_rank = rank
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_prebuilt=lambda: False))
    run = cls._pp_send_recv_and_preprocess_output_tensors
    output, batch_result, done, work = run(
        scheduler, 0, 0, [batch], ["metadata"], [], None
    )
    assert calls == (["send", "recv"] if rank == 0 else ["recv", "send"])
    assert output.tensors == {"value": "output"}
    assert (batch_result, done, work) == ("batch", event, ["work"])

    if new_api:
        calls.clear()
        state["skip"] = True
        output = run(scheduler, 0, 0, [batch], ["metadata"], [], None)
        assert calls == ["send"], "A skipped output must not post a receive"
        assert output == ("skipped", "batch", event, ["work"])
