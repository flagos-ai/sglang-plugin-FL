# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from sglang_fl.dispatch.backends.vendor.mthreads.lifecycle import (
    patch_pipeline_shutdown,
    patch_worker_http_shutdown,
)


@pytest.mark.parametrize("last_rank", [False, True])
def test_pp_forwards_shutdown_before_exiting_loop(monkeypatch, last_rank):
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    calls = []
    request = object()

    def loop(self):
        Scheduler.handle_shutdown(self, request)
        pytest.fail("PP must not receive another batch after ShutdownReq")

    def stop(self, req):
        assert req is request
        calls.append("stop")
        self.gracefully_exit = True

    def send(items, async_send):
        assert items == [request] and async_send
        calls.append("forward")
        return "shutdown-work"

    monkeypatch.setattr(SchedulerPPMixin, "event_loop_pp", loop)
    monkeypatch.setattr(Scheduler, "handle_shutdown", stop)
    patch_pipeline_shutdown()
    scheduler = SimpleNamespace(
        pp_group=SimpleNamespace(is_last_rank=last_rank),
        send_req_work="pending-work",
        _pp_commit_comm_work=lambda work: calls.append(work),
        _pp_send_pyobj_to_next_stage=send,
        gracefully_exit=False,
    )
    SchedulerPPMixin.event_loop_pp(scheduler)
    assert calls == (
        ["stop"] if last_rank else ["pending-work", "forward", "shutdown-work", "stop"]
    )
    assert scheduler.gracefully_exit
    assert not scheduler._musa_regular_pp_loop


def test_pp_does_not_hide_transport_failure(monkeypatch):
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    def send(*args, **kwargs):
        raise RuntimeError("broken peer")

    monkeypatch.setattr(
        SchedulerPPMixin,
        "event_loop_pp",
        lambda self: Scheduler.handle_shutdown(self, object()),
    )
    monkeypatch.setattr(
        Scheduler, "handle_shutdown", lambda *args: pytest.fail("not stopped")
    )
    patch_pipeline_shutdown()
    scheduler = SimpleNamespace(
        pp_group=SimpleNamespace(is_last_rank=False),
        send_req_work=[],
        _pp_commit_comm_work=lambda work: None,
        _pp_send_pyobj_to_next_stage=send,
    )
    with pytest.raises(RuntimeError, match="broken peer"):
        SchedulerPPMixin.event_loop_pp(scheduler)


def test_non_pp_shutdown_uses_original_handler(monkeypatch):
    from sglang.srt.managers.scheduler import Scheduler

    request = object()
    monkeypatch.setattr(Scheduler, "handle_shutdown", lambda self, req: req)
    patch_pipeline_shutdown()
    assert Scheduler.handle_shutdown(SimpleNamespace(), request) is request


@pytest.mark.parametrize(
    "node,tokenizer,expected",
    [(0, None, "http"), (1, None, None), (1, "tokenizer", "http")],
)
def test_worker_skips_only_absent_tokenizer_http(
    monkeypatch, node, tokenizer, expected
):
    from sglang.srt.entrypoints import http_server

    monkeypatch.setattr(
        http_server, "_setup_and_run_http_server", lambda *args, **kwargs: "http"
    )
    patch_worker_http_shutdown()
    assert (
        http_server._setup_and_run_http_server(
            SimpleNamespace(node_rank=node), tokenizer
        )
        == expected
    )


@pytest.mark.parametrize("exitcode", [0, 7, -9])
def test_worker_preserves_child_exit_failures(monkeypatch, exitcode):
    from sglang.srt.entrypoints.engine import Engine

    calls = []
    result = SimpleNamespace(block_until_scheduler_exits=lambda: calls.append("joined"))
    children = [SimpleNamespace(pid=123, exitcode=exitcode)]
    monkeypatch.setattr(
        Engine,
        "_launch_scheduler_processes",
        classmethod(lambda cls: (result, children)),
    )
    patch_worker_http_shutdown()
    actual, _ = Engine._launch_scheduler_processes()
    if exitcode:
        with pytest.raises(RuntimeError, match="Scheduler processes failed"):
            actual.block_until_scheduler_exits()
    else:
        actual.block_until_scheduler_exits()
    assert calls == ["joined"]
