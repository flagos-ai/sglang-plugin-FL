# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch PP send/recv ordering for txda platform.

On txda, point-to-point ``isend`` is effectively blocking (same as XPU).
If every PP rank calls send first, all ranks block waiting for a
receiver and the ring deadlocks. This patch replaces
``_pp_send_recv_and_preprocess_output_tensors`` with a version that
adds txda to the even/odd alternation check::

    send_first = (not is_xpu() and not is_txda()) or ((self.pp_rank % 2) == 0)
"""

import torch

from sglang.srt.distributed.parallel_state import P2PWork
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.utils.common import is_xpu

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._logger import patch_logger
from sglang_fl.dispatch.backends.vendor.tsingmicro.patches._utils import is_txda as _is_txda

_log = patch_logger("scheduler_pp_mixin")

_patched = False
_originals = {}


def patch() -> None:
    """Replace `_pp_send_recv_and_preprocess_output_tensors` to include txda.

    Idempotent: safe to call multiple times. No-op on non-txda platforms.
    """
    global _patched
    if _patched:
        return

    if not _is_txda():
        _log.skipped("txda not available - scheduler_pp_mixin patch skipped")
        return

    try:
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        _orig_fn = SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors
        _originals["_pp_send_recv_and_preprocess_output_tensors"] = _orig_fn

        # Replace the entire method - only the send_first line is changed.
        def _patched(
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
            # so every rank can send first safely. On some backends isend is
            # effectively blocking and does not return until the peer posts a
            # matching recv; if every PP rank sends first, all ranks block
            # waiting for a receiver and the ring deadlocks. Order send/recv
            # by pp_rank parity (even: send->recv, odd: recv->send) so each
            # adjacent pair has one sender and one receiver posted at the
            # same time.

            # CUDA: send first
            # XPU / TXDA: even ranks send first, odd ranks recv first.
            send_first = (not is_xpu() and not _is_txda()) or ((self.pp_rank % 2) == 0)

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
                        mbs[next_mb_id],
                        mb_metadata[next_mb_id],
                        next_pp_outputs,
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

        SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors = _patched
        _log.applied(
            "SchedulerPPMixin._pp_send_recv_and_preprocess_output_tensors patched"
            " -> txda uses even/odd PP send/recv ordering"
        )
    except Exception as exc:
        _log.failed(
            "failed to patch _pp_send_recv_and_preprocess_output_tensors: %s", exc
        )