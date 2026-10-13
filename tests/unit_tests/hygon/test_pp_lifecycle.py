# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""PP shutdown, worker exit errors, and FlagCX output ordering without a GPU."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from sglang_fl.dispatch.backends.vendor.hygon import lifecycle
from sglang_fl.dispatch.backends.vendor.hygon.patches.pp_output_launch_dependency import (
    _pp_send_dict_hook,
)


@pytest.fixture
def runtime(monkeypatch):
    names = (
        "sglang",
        "sglang.srt",
        "sglang.srt.managers",
        "sglang.srt.entrypoints",
        "sglang.srt.managers.scheduler",
        "sglang.srt.managers.scheduler_pp_mixin",
        "sglang.srt.entrypoints.http_server",
        "sglang.srt.entrypoints.engine",
    )
    modules = {}
    for name in names:
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)

    class Scheduler:
        def handle_shutdown(self, request):
            self.gracefully_exit = True

    class SchedulerPPMixin:
        def event_loop_pp(self):
            return Scheduler.handle_shutdown(self, self.request)

    class Engine:
        @classmethod
        def _launch_scheduler_processes(cls):
            return cls.result, cls.children

    modules["sglang.srt.managers.scheduler"].Scheduler = Scheduler
    modules[
        "sglang.srt.managers.scheduler_pp_mixin"
    ].SchedulerPPMixin = SchedulerPPMixin
    modules["sglang.srt.entrypoints.engine"].Engine = Engine
    http = modules["sglang.srt.entrypoints.http_server"]
    http._setup_and_run_http_server = lambda *args, **kwargs: "http"
    return SimpleNamespace(
        Scheduler=Scheduler, Mixin=SchedulerPPMixin, Engine=Engine, http=http
    )


@pytest.mark.parametrize("last_rank", [False, True])
def test_shutdown_forwarded_and_loop_unwinds(runtime, last_rank):
    lifecycle.patch_pipeline_shutdown()
    calls = []
    scheduler = SimpleNamespace(
        request=object(),
        pp_group=SimpleNamespace(is_last_rank=last_rank),
        send_req_work="pending",
        gracefully_exit=False,
        _pp_commit_comm_work=lambda work: calls.append(work),
        _pp_send_pyobj_to_next_stage=lambda requests, async_send: (
            calls.append(requests) or "stop-work"
        ),
    )
    runtime.Mixin.event_loop_pp(scheduler)
    assert scheduler.gracefully_exit
    assert not scheduler._hcu_regular_pp_loop
    assert calls == ([] if last_rank else ["pending", [scheduler.request], "stop-work"])


def test_shutdown_does_not_hide_transport_error(runtime):
    lifecycle.patch_pipeline_shutdown()

    def fail(*args, **kwargs):
        raise RuntimeError("broken peer")

    scheduler = SimpleNamespace(
        request=object(),
        pp_group=SimpleNamespace(is_last_rank=False),
        send_req_work=[],
        _pp_commit_comm_work=lambda work: None,
        _pp_send_pyobj_to_next_stage=fail,
        gracefully_exit=False,
    )
    with pytest.raises(RuntimeError, match="broken peer"):
        runtime.Mixin.event_loop_pp(scheduler)
    assert not scheduler.gracefully_exit
    assert not scheduler._hcu_regular_pp_loop


@pytest.mark.parametrize(
    "node,tokenizer,expected",
    [(0, None, "http"), (1, None, None), (1, object(), "http")],
)
def test_worker_http_requires_tokenizer(runtime, node, tokenizer, expected):
    lifecycle.patch_worker_http_shutdown()
    assert (
        runtime.http._setup_and_run_http_server(
            SimpleNamespace(node_rank=node), tokenizer
        )
        == expected
    )


@pytest.mark.parametrize("exitcode", [0, 7, -9])
def test_worker_failure_propagates(runtime, exitcode):
    calls = []
    runtime.Engine.result = SimpleNamespace(
        block_until_scheduler_exits=lambda: calls.append("joined")
    )
    runtime.Engine.children = [SimpleNamespace(pid=42, exitcode=exitcode)]
    lifecycle.patch_worker_http_shutdown()
    result, _ = runtime.Engine._launch_scheduler_processes()
    if exitcode:
        with pytest.raises(RuntimeError, match="Scheduler processes failed"):
            result.block_until_scheduler_exits()
    else:
        result.block_until_scheduler_exits()
    assert calls == ["joined"]


@pytest.mark.parametrize(
    "active,last,msg,waits",
    [
        (True, False, "output", 1),
        (True, True, "output", 0),
        (False, False, "output", 0),
        (True, False, "proxy", 0),
    ],
)
def test_flagcx_output_waits_for_model_launch(active, last, msg, waits):
    events = []
    event = object()
    scheduler = SimpleNamespace(
        pp_group=SimpleNamespace(
            is_last_rank=last,
            fl_communicator=SimpleNamespace(
                _flagcx_comm=SimpleNamespace(disabled=not active)
            ),
        ),
        launch_event=event,
        device="cuda",
        device_module=SimpleNamespace(
            current_stream=lambda **kwargs: SimpleNamespace(wait_event=events.append)
        ),
    )
    result = _pp_send_dict_hook(
        lambda self, *args, **kwargs: "sent", scheduler, {}, True, msg
    )
    assert result == "sent"
    assert events == [event] * waits
