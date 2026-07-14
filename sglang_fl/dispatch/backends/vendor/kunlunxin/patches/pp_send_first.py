"""Kunlunxin: order PP ring send/recv by first-rank instead of pp_rank parity.

``SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors`` decides, for
each pipeline-parallel step, whether the rank sends its output to the next stage
before or after receiving from the previous stage. On backends where ``isend``
is effectively blocking, "everyone sends first" deadlocks the ring, so the order
must be staggered between adjacent ranks.

Upstream staggers by ``pp_rank`` parity (even ranks send first, odd ranks recv
first) on XPU. On Kunlunxin we instead key the decision off the PP group's
first-rank flag:

    send_first = self.pp_group.is_first_rank

Only the first rank sends before receiving; every downstream rank receives
first. The rest of the method is identical to upstream — we replace the whole
method because ``send_first`` is a local variable (not a module global), so it
cannot be rebound the way clamp_position patches module attributes.
"""

import logging

logger = logging.getLogger(__name__)
_applied = False


def _pp_send_recv_and_preprocess_output_tensors(
    self,
    next_first_rank_mb_id,
    next_mb_id,
    mbs,
    mb_metadata,
    last_rank_comm_queue,
    pp_outputs,
):
    import torch
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

    next_pp_outputs = None
    d2h_event = None
    batch_result = None
    send_output_work = []

    # Kunlunxin: order the PP ring by first-rank instead of pp_rank parity —
    # only the first rank sends before receiving; all others recv first.
    send_first = self.pp_group.is_first_rank

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


def patch_pp_send_recv_and_preprocess_output_tensors():
    """Rebind SchedulerPPMixin PP send/recv ordering to first-rank on klx."""
    global _applied
    if _applied:
        return

    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors = (
        _pp_send_recv_and_preprocess_output_tensors
    )

    _applied = True
    logger.info(
        "patched SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors "
        "-> send_first = pp_group.is_first_rank (Kunlunxin PP ring ordering)"
    )
