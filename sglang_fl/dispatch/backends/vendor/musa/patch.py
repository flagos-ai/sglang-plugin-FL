from __future__ import annotations

import logging
from functools import wraps

logger = logging.getLogger(__name__)

_patches_applied = False


def _patch_pp_send_recv_order() -> None:
    try:
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
    except Exception as e:
        logger.warning("MUSA PP send/recv order patch skipped: %s", e)
        return

    import torch

    orig_fn = SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors

    @wraps(orig_fn)
    def pp_send_recv_and_preprocess_output_tensors_with_musa_order(
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

        if (self.pp_rank % 2) == 0:
            send_output_work = _do_send()
            _do_recv()
        else:
            _do_recv()
            send_output_work = _do_send()

        return next_pp_outputs, batch_result, d2h_event, send_output_work

    SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors = (
        pp_send_recv_and_preprocess_output_tensors_with_musa_order
    )
    logger.info("MUSA PP send/recv ordering patch applied")


def _patch_pp_launch_batch_add_sync() -> None:
    try:
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
    except Exception as e:
        logger.warning("MUSA PP launch sync patch skipped: %s", e)
        return

    orig_fn = SchedulerPPMixin._pp_launch_batch

    @wraps(orig_fn)
    def pp_launch_batch_with_forward_stream_sync(self, *args, **kwargs):
        result, event = orig_fn(self, *args, **kwargs)
        self.forward_stream.synchronize()
        return result, event

    SchedulerPPMixin._pp_launch_batch = pp_launch_batch_with_forward_stream_sync
    logger.info("MUSA PP launch forward_stream sync patch applied")

def apply_musa_patches() -> None:
    global _patches_applied
    if _patches_applied:
        return

    _patch_pp_send_recv_order()
    _patch_pp_launch_batch_add_sync()
    _patches_applied = True
    logger.info("All MUSA PP patches applied successfully")


apply_musa_patches()
