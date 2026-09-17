# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SGLang 0.5.18 worker shutdown compatibility for MUSA."""

import logging
from functools import wraps

logger = logging.getLogger(__name__)


class _PipelineShutdown(BaseException):
    """Unwind only a regular PP loop after its explicit stop request."""


def patch_pipeline_shutdown() -> None:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    if not hasattr(Scheduler, "handle_shutdown"):
        return
    original_loop = SchedulerPPMixin.event_loop_pp
    original_shutdown = Scheduler.handle_shutdown

    @wraps(original_loop)
    def event_loop_pp(self, *args, **kwargs):
        self._musa_regular_pp_loop = True
        try:
            return original_loop(self, *args, **kwargs)
        except _PipelineShutdown:
            return None
        finally:
            self._musa_regular_pp_loop = False

    @wraps(original_shutdown)
    def handle_shutdown(self, request):
        if not getattr(self, "_musa_regular_pp_loop", False):
            return original_shutdown(self, request)
        # The upstream PP loop has no gracefully_exit check. Forward the stop
        # request before unwinding, so later stages can also exit in userspace.
        # Transport failures still propagate through the normal error path.
        if not self.pp_group.is_last_rank:
            self._pp_commit_comm_work(self.send_req_work)
            work = self._pp_send_pyobj_to_next_stage([request], async_send=True)
            self._pp_commit_comm_work(work)
        original_shutdown(self, request)
        raise _PipelineShutdown()

    SchedulerPPMixin.event_loop_pp = event_loop_pp
    Scheduler.handle_shutdown = handle_shutdown


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


def apply_musa_lifecycle_patches() -> None:
    patch_pipeline_shutdown()
    patch_worker_http_shutdown()
    logger.info("MUSA explicit PP shutdown and worker HTTP exit patches applied")
