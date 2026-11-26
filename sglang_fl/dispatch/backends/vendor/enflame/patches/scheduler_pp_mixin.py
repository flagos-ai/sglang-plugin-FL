from collections import deque
from typing import List, Optional

import torch

from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.managers.scheduler_pp_mixin import (
    PPBatchMetadata,
    _pp_can_skip_output_comm,
)

from sglang.srt.managers.scheduler import Scheduler

def _pp_launch_batch(
        self: Scheduler,
        mb_id: int,
        cur_batch,
        pp_proxy_tensors: PPProxyTensors,
        mb_metadata: List[Optional[PPBatchMetadata]],
        last_rank_comm_queue: deque,
    ):
        with torch.profiler.record_function("run_batch"):
            with self.forward_stream_ctx:
                self.forward_stream.wait_stream(self.schedule_stream)
                set_time_batch(
                    cur_batch.reqs,
                    "set_run_batch_cpu_start_time",
                    trace_only=True,
                )
                result = self.run_batch(cur_batch, pp_proxy_tensors)
                # GCU limits
                major, _ = torch.gcu.get_device_capability("gcu")
                if major == 3:
                    torch.cuda.synchronize() # add sync, before next batch
                set_time_batch(
                    cur_batch.reqs,
                    "set_run_batch_cpu_end_time",
                    trace_only=True,
                    attrs={"pp_mb_id": mb_id},
                )
                mb_metadata[mb_id] = PPBatchMetadata(
                    can_run_cuda_graph=result.can_run_cuda_graph,
                )
                event = self.device_module.Event()
                event.record(self.device_module.current_stream())
                if self.pp_group.is_last_rank:
                    # (last rank) buffer the outputs for async batch depth
                    last_rank_comm_queue.append(
                        (
                            event,
                            PPProxyTensors(
                                self._pp_prepare_tensor_dict(result, cur_batch)
                            ),
                        )
                    )
        return result, event

# Derived from SGLang v0.5.18; licensed under Apache-2.0.
def _pp_send_recv_and_preprocess_output_tensors(
    self: Scheduler,
    next_first_rank_mb_id: int,
    next_mb_id: int,
    mbs: List[Optional[object]],
    mb_metadata: List[Optional[PPBatchMetadata]],
    last_rank_comm_queue: deque,
    pp_outputs,
):
    """Alternate the send/recv order by pipeline rank on GCU.

    Upstream derives the order from ``is_xpu()``::

        send_first = (not is_xpu()) or ((self.ps.pp_rank % 2) == 0)

    ECCL point-to-point work is matched by posting order rather than by tag,
    so on GCU every rank must not post its sends first.  Doing so lets the
    earlier receives consume the sends that later receives are waiting for,
    and both peers end up blocked inside an unmatched receive (observed as
    every pipeline rank's ``ecclKernel_SendRecv_Normal`` stuck in ``run``
    while the schedulers spin in ``d2h_event.synchronize()``).  Keep the
    even-send-first ring ordering that the XPU path uses.
    """
    next_pp_outputs = None
    d2h_event = None
    batch_result = None
    send_output_work = []

    send_first = (self.ps.pp_rank % 2) == 0

    def _do_send():
        return self._pp_send_output_to_next_stage(
            next_first_rank_mb_id,
            mbs,
            last_rank_comm_queue,
            pp_outputs,
        )

    def _do_recv():
        nonlocal next_pp_outputs, batch_result, d2h_event
        target = mbs[next_mb_id]
        if target is None or target.forward_mode.is_prebuilt():
            return
        if _pp_can_skip_output_comm(target):
            next_pp_outputs, batch_result, d2h_event = (
                self._pp_make_skip_output_result(target, mb_metadata[next_mb_id])
            )
            return
        with torch.profiler.record_function("recv_res_dict_from_prev_stage"):
            next_pp_outputs = PPProxyTensors(self._pp_recv_dict_from_prev_stage())
        with self.copy_stream_ctx:
            self.copy_stream.wait_stream(self.schedule_stream)
            batch_result = self._pp_prep_batch_result(
                target, mb_metadata[next_mb_id], next_pp_outputs
            )
            d2h_event = self.device_module.Event()
            d2h_event.record(self.device_module.current_stream())
    if send_first:
        send_output_work = _do_send()
        _do_recv()
    else:
        _do_recv()
        send_output_work = _do_send()

    return next_pp_outputs, batch_result, d2h_event, send_output_work


def _gcu_pp_init_loop_state(original_fn, self, *args, **kwargs):
    """GCU: disable the PP send-allgather optimisation.

    ``require_attn_tp_allgather`` selects whether PP tensor-dict send/recv goes
    through the "send-allgather" path, which strips the tensor per rank on the
    send side and rebuilds it with an attention-TP-group all-gather on the recv
    side (``CommunicatorFL.recv_tensor_dict``).  That all-gather is an
    order-matched collective: every rank of the group must enter the same call.
    But ``use_all_gather`` is decided independently per rank from the metadata
    that rank received (``tensor.numel() % all_gather_size == 0``, empty tensors
    are skipped), so a rank that gets an empty or absent entry skips the
    collective while its peers block in it and never retire their stream --
    the PP pipeline deadlocks.

    The flag is set once in ``init_pp_loop_state`` and used only to build the
    ``all_gather_group`` argument at the PP send/recv call sites, so clearing it
    makes both sides fall back to plain point-to-point full-tensor transfer:
    symmetric, no group collective, cannot desynchronise.  Cost: more PP traffic.
    """

    result = original_fn(self, *args, **kwargs)
    self.require_attn_tp_allgather = False
    return result


def patch_scheduler_pp_mixin():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    SCHEDULER_PP_MIXIN = "sglang.srt.managers.scheduler_pp_mixin.SchedulerPPMixin"
    HookRegistry.register(
        f"{SCHEDULER_PP_MIXIN}._pp_launch_batch", _pp_launch_batch, HookType.REPLACE
    )
    HookRegistry.register(
        f"{SCHEDULER_PP_MIXIN}._pp_send_recv_and_preprocess_output_tensors",
        _pp_send_recv_and_preprocess_output_tensors,
        HookType.REPLACE,
    )
    HookRegistry.register(
        f"{SCHEDULER_PP_MIXIN}.init_pp_loop_state",
        _gcu_pp_init_loop_state,
        HookType.AROUND,
    )
