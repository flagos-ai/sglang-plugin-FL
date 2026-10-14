# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SGLang 0.5.18 worker shutdown compatibility for Hygon."""

import logging
from functools import wraps

logger = logging.getLogger(__name__)


class _PipelineShutdown(BaseException):
    """Unwind a regular PP loop only after every rank drained its work."""


def patch_pipeline_shutdown() -> None:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    if not hasattr(Scheduler, "handle_shutdown"):
        return
    original_loop = SchedulerPPMixin.event_loop_pp
    original_shutdown = Scheduler.handle_shutdown
    original_idle = Scheduler.on_idle

    @wraps(original_loop)
    def event_loop_pp(self, *args, **kwargs):
        self._hcu_regular_pp_loop = True
        self._hcu_pending_pp_shutdown = None
        try:
            return original_loop(self, *args, **kwargs)
        except _PipelineShutdown:
            return None
        finally:
            self._hcu_regular_pp_loop = False
            self._hcu_pending_pp_shutdown = None

    @wraps(original_shutdown)
    def handle_shutdown(self, request):
        if not getattr(self, "_hcu_regular_pp_loop", False):
            return original_shutdown(self, request)
        # The normal PP loop forwards this request to the next stage. Keep
        # running that loop: an earlier stage still owes its peers the output
        # of the previous microbatch even after the client received its answer.
        # Do not set gracefully_exit yet; the shared get_next_batch hook would
        # otherwise exit before that output send/recv can finish.
        self._hcu_pending_pp_shutdown = request
        return None

    @wraps(original_idle)
    def on_idle(self, *args, **kwargs):
        request = getattr(self, "_hcu_pending_pp_shutdown", None)
        if not getattr(self, "_hcu_regular_pp_loop", False) or request is None:
            return original_idle(self, *args, **kwargs)
        # Upstream calls on_idle only after a whole PP round has no runnable
        # batch. Its idle predicate additionally checks every microbatch slot.
        # Wait for buffered output metadata too; sleeping here could prevent
        # the next round from forwarding the remaining output to a peer.
        if (
            not self.is_fully_idle()
            or self.pp_outputs is not None
            or self.last_rank_comm_queue
            or any(self._pp_tensor_dict_inbox.values())
        ):
            return None
        for work in (
            self.send_req_work,
            self.send_proxy_work,
            self.send_output_work,
        ):
            self._pp_commit_comm_work(work)
        # CPU metadata completion does not wait for FlagCX GPU P2P streams.
        self.device_module.synchronize()
        # A fast TP rank must retain its groups while slower PP/TP peers finish
        # their last receives. Every scheduler participates in this CPU group.
        from torch.distributed import barrier

        barrier(group=self.world_group.cpu_group)
        original_shutdown(self, request)
        logger.info("HCU PP shutdown drained and synchronized across all ranks")
        raise _PipelineShutdown()

    SchedulerPPMixin.event_loop_pp = event_loop_pp
    Scheduler.handle_shutdown = handle_shutdown
    Scheduler.on_idle = on_idle


def patch_worker_http_shutdown() -> None:
    from sglang.srt.entrypoints import http_server
    from sglang.srt.entrypoints.engine import Engine

    if not hasattr(http_server, "_setup_and_run_http_server"):
        return
    original_http = http_server._setup_and_run_http_server
    original_launch = Engine._launch_scheduler_processes.__func__

    @wraps(original_http)
    def setup_http(server_args, tokenizer_manager, *args, **kwargs):
        if server_args.node_rank > 0 and tokenizer_manager is None:
            # _launch_subprocesses has already waited for this node's workers.
            # Nonzero nodes have no tokenizer and cannot initialize HTTP serving.
            return None
        return original_http(server_args, tokenizer_manager, *args, **kwargs)

    @wraps(original_launch)
    def launch_schedulers(cls, *args, **kwargs):
        result, processes = original_launch(cls, *args, **kwargs)
        if processes is not None:
            original_wait = result.block_until_scheduler_exits

            def wait_for_clean_exit():
                original_wait()
                failures = [(p.pid, p.exitcode) for p in processes if p.exitcode != 0]
                if failures:
                    raise RuntimeError(f"Scheduler processes failed: {failures}")

            result.block_until_scheduler_exits = wait_for_clean_exit
        return result, processes

    http_server._setup_and_run_http_server = setup_http
    Engine._launch_scheduler_processes = classmethod(launch_schedulers)


def apply_hcu_lifecycle_patches() -> None:
    patch_pipeline_shutdown()
    patch_worker_http_shutdown()
    logger.info("HCU explicit PP shutdown and worker HTTP exit patches applied")
