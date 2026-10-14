# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""PP drain/exit protocol and FlagCX output ordering without a GPU."""

import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
import torch.distributed as dist

import sglang_fl
from sglang_fl.dispatch.backends.vendor.hygon import lifecycle
from sglang_fl.dispatch.backends.vendor.hygon.patches.pp_output_launch_dependency import (
    _pp_send_dict_hook,
)


@pytest.fixture
def runtime(monkeypatch):
    names = (
        "sglang", "sglang.srt", "sglang.srt.managers", "sglang.srt.entrypoints",
        "sglang.srt.plugins", "sglang.srt.plugins.hook_registry",
        "sglang.srt.managers.scheduler", "sglang.srt.managers.scheduler_pp_mixin",
        "sglang.srt.entrypoints.http_server", "sglang.srt.entrypoints.engine",
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
            self.events.append("mark-graceful")
            self.gracefully_exit = True

        def on_idle(self):
            self.events.append("idle-housekeeping")
            return "housekeeping"

    class SchedulerPPMixin:
        def event_loop_pp(self):
            return self.run_pipeline(self)

    class Engine:
        @classmethod
        def _launch_scheduler_processes(cls):
            return cls.result, cls.children

    hooks = {}

    class HookRegistry:
        @staticmethod
        def register(target, hook, hook_type):
            hooks[target] = hook

    hook_module = modules["sglang.srt.plugins.hook_registry"]
    hook_module.HookRegistry = HookRegistry
    hook_module.HookType = SimpleNamespace(AROUND="around")
    # Exercise the actual shared early-exit hook, rather than duplicating it.
    sglang_fl._setup_communicator_hooks()
    barrier_calls = []
    monkeypatch.setattr(dist, "barrier", lambda *, group: barrier_calls.append(group))
    modules["sglang.srt.managers.scheduler"].Scheduler = Scheduler
    modules["sglang.srt.managers.scheduler_pp_mixin"].SchedulerPPMixin = SchedulerPPMixin
    modules["sglang.srt.entrypoints.engine"].Engine = Engine
    http = modules["sglang.srt.entrypoints.http_server"]
    http._setup_and_run_http_server = lambda *args, **kwargs: "http"
    return SimpleNamespace(
        Scheduler=Scheduler, Mixin=SchedulerPPMixin, Engine=Engine, http=http,
        batch_hook=hooks["sglang.srt.managers.scheduler.Scheduler.get_next_batch_to_run"],
        barrier_calls=barrier_calls,
    )


def scheduler_state():
    events = []

    def commit(work):
        events.append(("commit", tuple(work)))
        work.clear()

    return SimpleNamespace(
        events=events, request=object(), gracefully_exit=False,
        ps=SimpleNamespace(pp_size=2), world_group=SimpleNamespace(cpu_group=object()),
        send_req_work=["request"], send_proxy_work=["proxy"], send_output_work=["output"],
        pp_outputs=None, last_rank_comm_queue=[], _pp_tensor_dict_inbox={},
        is_fully_idle=lambda: True, _pp_commit_comm_work=commit,
        device_module=SimpleNamespace(synchronize=lambda: events.append("gpu-sync")),
    )


def test_shutdown_keeps_shared_batch_hook_running_until_tail_drained(runtime):
    lifecycle.patch_pipeline_shutdown()
    scheduler = scheduler_state()

    def pipeline(self):
        runtime.Scheduler.handle_shutdown(self, self.request)
        assert not self.gracefully_exit
        assert self._hcu_pending_pp_shutdown is self.request
        assert runtime.batch_hook(lambda self: "continue-pp", self) == "continue-pp"
        # The normal loop, not the shutdown handler, forwards the request and
        # forwards the previous microbatch's output before reaching on_idle.
        self.events.extend(["normal-request-forward", "tail-output-forward"])
        return runtime.Scheduler.on_idle(self)

    scheduler.run_pipeline = pipeline
    runtime.Mixin.event_loop_pp(scheduler)
    assert scheduler.gracefully_exit
    assert not scheduler._hcu_regular_pp_loop
    assert scheduler._hcu_pending_pp_shutdown is None
    assert runtime.barrier_calls == [scheduler.world_group.cpu_group]
    assert scheduler.events == [
        "normal-request-forward", "tail-output-forward", ("commit", ("request",)),
        ("commit", ("proxy",)), ("commit", ("output",)), "gpu-sync", "mark-graceful",
    ]


@pytest.mark.parametrize("busy", ["microbatches", "pp_outputs", "last_rank", "inbox"])
def test_shutdown_waits_for_microbatches_and_buffered_communication(runtime, busy):
    lifecycle.patch_pipeline_shutdown()
    scheduler = scheduler_state()
    if busy == "microbatches":
        scheduler.is_fully_idle = lambda: False
    elif busy == "pp_outputs":
        scheduler.pp_outputs = object()
    elif busy == "last_rank":
        scheduler.last_rank_comm_queue.append(object())
    else:
        scheduler._pp_tensor_dict_inbox["output"] = [object()]

    def pipeline(self):
        runtime.Scheduler.handle_shutdown(self, self.request)
        assert runtime.Scheduler.on_idle(self) is None
        assert self.events == []
        assert not self.gracefully_exit
        assert runtime.barrier_calls == []
        assert runtime.batch_hook(lambda self: "drain-next-slot", self) == "drain-next-slot"
        self.is_fully_idle = lambda: True
        self.pp_outputs = None
        self.last_rank_comm_queue.clear()
        self._pp_tensor_dict_inbox.clear()
        runtime.Scheduler.on_idle(self)

    scheduler.run_pipeline = pipeline
    runtime.Mixin.event_loop_pp(scheduler)
    assert scheduler.gracefully_exit
    assert runtime.barrier_calls == [scheduler.world_group.cpu_group]


@pytest.mark.parametrize("failure", ["request", "proxy", "output", "gpu", "barrier"])
def test_shutdown_does_not_hide_transport_or_synchronization_errors(runtime, monkeypatch, failure):
    lifecycle.patch_pipeline_shutdown()
    scheduler = scheduler_state()
    original_commit = scheduler._pp_commit_comm_work

    def fail():
        raise RuntimeError(f"broken {failure}")

    def commit(work):
        if work == [failure]:
            fail()
        original_commit(work)

    scheduler._pp_commit_comm_work = commit
    if failure == "gpu":
        scheduler.device_module.synchronize = fail
    elif failure == "barrier":
        monkeypatch.setattr(dist, "barrier", lambda **kwargs: fail())

    def pipeline(self):
        runtime.Scheduler.handle_shutdown(self, self.request)
        runtime.Scheduler.on_idle(self)

    scheduler.run_pipeline = pipeline
    with pytest.raises(RuntimeError, match=f"broken {failure}"):
        runtime.Mixin.event_loop_pp(scheduler)
    assert not scheduler.gracefully_exit
    assert not scheduler._hcu_regular_pp_loop
    assert scheduler._hcu_pending_pp_shutdown is None


def test_non_pp_shutdown_and_normal_idle_keep_upstream_behavior(runtime):
    lifecycle.patch_pipeline_shutdown()
    scheduler = scheduler_state()
    assert runtime.Scheduler.on_idle(scheduler) == "housekeeping"
    runtime.Scheduler.handle_shutdown(scheduler, scheduler.request)
    assert scheduler.gracefully_exit
    assert runtime.barrier_calls == []
    assert scheduler.events == ["idle-housekeeping", "mark-graceful"]


@pytest.mark.parametrize("slow_rank", [2, 3])
def test_four_ranks_keep_peers_alive_for_tail_output_and_stop_broadcast(runtime, monkeypatch, slow_rank):
    """Model the real PP one-slot output lag with a delayed last-stage TP rank.

    First-stage TP1 receives ShutdownReq while last-stage TP1 still needs its
    previous output. Closing there reproduced the physical test's Gloo error.
    The normal PP protocol must deliver that tail, broadcast stop to both last-
    stage TP ranks, complete work and GPU sync, then let all four ranks exit.
    """
    lifecycle.patch_pipeline_shutdown()
    world_group = object()
    world_barrier = threading.Barrier(4)
    stage1_broadcast = threading.Barrier(2)
    stop_forwarded = threading.Event()
    stop_received = threading.Event()
    tail_sent = [threading.Event(), threading.Event()]
    tail_received = [threading.Event(), threading.Event()]
    release_slow_peer = threading.Event()
    closed = [False] * 4
    arrivals = []
    failures = []
    local = threading.local()
    lock = threading.Lock()
    schedulers = [scheduler_state() for _ in range(4)]

    def wait(event, label):
        if not event.wait(timeout=3):
            raise RuntimeError(f"Unmatched {label}")

    def barrier(*, group):
        assert group is world_group
        with lock:
            assert not any(closed), "A scheduler closed before all peers drained"
            arrivals.append(local.rank)
        world_barrier.wait(timeout=3)

    monkeypatch.setattr(dist, "barrier", barrier)

    class Work:
        def __init__(self, event, label):
            self.event, self.label = event, label

        def wait(self):
            wait(self.event, self.label)

    def drive(rank):
        local.rank = rank
        self = schedulers[rank]
        tp_rank = rank % 2
        self.world_group.cpu_group = world_group
        self.send_req_work = []
        self.send_proxy_work = []
        self.send_output_work = []
        if rank == 0:
            self.send_req_work = [SimpleNamespace(work=Work(stop_received, "stop request broadcast"))]
        if rank < 2:
            self.send_output_work = [SimpleNamespace(work=Work(tail_received[tp_rank], "tail output recv"))]

        def commit(work):
            for item in work:
                item.work.wait()
            work.clear()

        self._pp_commit_comm_work = commit

        def gpu_sync():
            assert tail_received[tp_rank].is_set()
            self.events.append("gpu-sync")

        self.device_module.synchronize = gpu_sync

        def pipeline(scheduler):
            if rank == slow_rank:
                wait(release_slow_peer, "test peer release")
            if rank >= 2:
                # A PP rank may still be receiving the prior slot's result when
                # its TP leader first receives the next slot's ShutdownReq.
                wait(tail_sent[tp_rank], "previous output send")
                tail_received[tp_rank].set()
                wait(stop_forwarded, "forwarded stop")
                stage1_broadcast.wait(timeout=3)
            runtime.Scheduler.handle_shutdown(scheduler, scheduler.request)
            assert not scheduler.gracefully_exit
            assert runtime.batch_hook(lambda self: "next-slot", scheduler) == "next-slot"
            if rank == 0:
                stop_forwarded.set()  # normal asynchronous request forwarding
            if rank < 2:
                tail_sent[tp_rank].set()  # normal output step after the handler
            else:
                stop_received.set()
            runtime.Scheduler.on_idle(scheduler)

        self.run_pipeline = pipeline
        try:
            runtime.Mixin.event_loop_pp(self)
            closed[rank] = True
        except BaseException as error:
            with lock:
                failures.append((rank, repr(error)))

    threads = [threading.Thread(target=drive, args=(rank,), daemon=True) for rank in range(4)]
    for thread in threads:
        thread.start()
    try:
        assert tail_sent[1].wait(timeout=3), "TP1 exited before forwarding its prior output"
        assert not any(closed)
    finally:
        release_slow_peer.set()
        for thread in threads:
            thread.join(timeout=4)
    assert not any(thread.is_alive() for thread in threads)
    assert failures == []
    assert sorted(arrivals) == [0, 1, 2, 3]
    assert closed == [True] * 4
    assert all(scheduler.gracefully_exit for scheduler in schedulers)
    assert all(scheduler.events == ["gpu-sync", "mark-graceful"] for scheduler in schedulers)


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
