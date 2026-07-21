"""Ascend/NPU patches for SGLang pipeline-parallel scheduler mixin.

Ports two source edits from
``sglang/srt/managers/scheduler_pp_mixin.py`` into the plugin patch layer:

  1. ``_pp_send_recv_and_preprocess_output_tensors``: HCCL isend is
     effectively blocking (unlike CUDA), so if every PP rank sends first the
     ring deadlocks. Order send/recv by ``pp_rank`` parity (even: send->recv,
     odd: recv->send) and synchronize the device before the exchange.
  2. ``_pp_launch_batch``: synchronize ``forward_stream`` before returning so
     the forward computation is complete before the PP send/recv runs.
"""

from __future__ import annotations

import logging
from functools import wraps

logger = logging.getLogger(__name__)


def patch_pp_send_recv_order() -> None:
    try:
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
    except Exception as e:
        logger.warning("Ascend PP send/recv order patch skipped: %s", e)
        return

    import torch

    def _pp_send_recv_and_preprocess_output_tensors(
        self,
        next_first_rank_mb_id,
        next_mb_id,
        mbs,
        mb_metadata,
        last_rank_comm_queue,
        pp_outputs,
    ):
        next_pp_outputs = None
        d2h_event = None
        batch_result = None
        send_output_work = []

        # On CUDA, isend is async: it enqueues to the stream and returns,
        # so every rank can send first safely. On HCCL isend is effectively
        # blocking and does not return until the peer posts a matching recv;
        # if every PP rank sends first, all ranks block waiting for a receiver
        # and the ring deadlocks. Order send/recv by pp_rank parity (even:
        # send->recv, odd: recv->send) so each adjacent pair has one sender and
        # one receiver posted at the same time.
        send_first = (self.pp_rank % 2) == 0
        if self.device == "npu":
            self.device_module.synchronize()

        def _do_send():
            return self._pp_send_output_to_next_stage(
                next_first_rank_mb_id,
                mbs,
                last_rank_comm_queue,
                pp_outputs,
            )

        def _do_recv():
            nonlocal next_pp_outputs, batch_result, d2h_event
            if mbs[next_mb_id] is None or mbs[next_mb_id].forward_mode.is_prebuilt():
                return
            with torch.profiler.record_function("recv_res_dict_from_prev_stage"):
                next_pp_outputs = PPProxyTensors(self._pp_recv_dict_from_prev_stage())
            with self.copy_stream_ctx:
                self.copy_stream.wait_stream(self.schedule_stream)
                batch_result = self._pp_prep_batch_result(
                    mbs[next_mb_id], mb_metadata[next_mb_id], next_pp_outputs
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

    SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors = (
        _pp_send_recv_and_preprocess_output_tensors
    )
    logger.info("Ascend PP send/recv ordering patch applied")


def patch_pp_launch_batch_sync() -> None:
    try:
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
    except Exception as e:
        logger.warning("Ascend PP launch sync patch skipped: %s", e)
        return

    orig_fn = SchedulerPPMixin._pp_launch_batch

    @wraps(orig_fn)
    def _pp_launch_batch_with_forward_stream_sync(self, *args, **kwargs):
        result, event = orig_fn(self, *args, **kwargs)
        # NPU fix: ensure forward computation completes before PP send/recv
        self.forward_stream.synchronize()
        return result, event

    SchedulerPPMixin._pp_launch_batch = _pp_launch_batch_with_forward_stream_sync
    logger.info("Ascend PP launch forward_stream sync patch applied")
