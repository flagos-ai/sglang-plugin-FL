"""Ascend pipeline-parallel compatibility for SGLang 0.5.18.

HCCL point-to-point sends can wait until the peer posts a receive. SGLang
0.5.18 already contains a parity-ordered exchange for XPU, so this patch keeps
the upstream implementation intact and enables that same branch for Ascend.
It also synchronizes NPU work at the two boundaries required before the
pipeline exchange. This deliberately avoids copying the scheduler body: new
0.5.18 behavior such as skip-output communication remains owned by SGLang.
"""

from __future__ import annotations

import inspect
import logging
from functools import wraps

logger = logging.getLogger(__name__)

_SEND_RECV_PARAMETERS = (
    "self",
    "next_first_rank_mb_id",
    "next_mb_id",
    "mbs",
    "mb_metadata",
    "last_rank_comm_queue",
    "pp_outputs",
)
_LAUNCH_PARAMETERS = (
    "self",
    "mb_id",
    "cur_batch",
    "pp_proxy_tensors",
    "mb_metadata",
    "last_rank_comm_queue",
)


def _require_signature(function, expected: tuple[str, ...], label: str) -> None:
    actual = tuple(inspect.signature(function).parameters)
    if actual != expected:
        raise RuntimeError(
            "Unsupported SGLang scheduler interface for the Ascend 0.5.18 "
            f"patch: {label}{actual}, expected {expected}."
        )


def patch_pp_send_recv_order() -> None:
    """Enable upstream's parity ordering for HCCL and sync before exchange."""

    try:
        from sglang.srt.managers import scheduler_pp_mixin as pp_module
    except ImportError as exc:
        raise RuntimeError(
            "SGLang 0.5.18 scheduler_pp_mixin is required by the Ascend backend"
        ) from exc

    mixin = pp_module.SchedulerPPMixin
    original = mixin._pp_send_recv_and_preprocess_output_tensors
    if getattr(original, "_sglang_fl_ascend_ordered", False):
        return

    _require_signature(
        original,
        _SEND_RECV_PARAMETERS,
        "_pp_send_recv_and_preprocess_output_tensors",
    )
    if not hasattr(pp_module, "is_xpu"):
        raise RuntimeError(
            "SGLang 0.5.18 scheduler no longer exposes the parity-order hook "
            "expected by the Ascend backend"
        )

    # The upstream function consults this module global at exactly one point:
    # `send_first = (not is_xpu()) or pp_rank % 2 == 0`. An Ascend worker owns
    # its process, so selecting the ordered branch here cannot affect another
    # device backend and retains the rest of the upstream function verbatim.
    if not hasattr(pp_module, "_sglang_fl_native_is_xpu"):
        pp_module._sglang_fl_native_is_xpu = pp_module.is_xpu

    def _requires_ordered_pp_transport() -> bool:
        return True

    pp_module.is_xpu = _requires_ordered_pp_transport

    @wraps(original)
    def _send_recv_with_npu_sync(self, *args, **kwargs):
        self.device_module.synchronize()
        return original(self, *args, **kwargs)

    _send_recv_with_npu_sync._sglang_fl_ascend_ordered = True
    _send_recv_with_npu_sync._sglang_fl_original = original
    mixin._pp_send_recv_and_preprocess_output_tensors = _send_recv_with_npu_sync
    logger.info("Ascend PP parity ordering and pre-exchange sync applied")


def patch_pp_launch_batch_sync() -> None:
    """Wait for the forward stream before a microbatch enters PP exchange."""

    try:
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
    except ImportError as exc:
        raise RuntimeError(
            "SGLang 0.5.18 scheduler_pp_mixin is required by the Ascend backend"
        ) from exc

    original = SchedulerPPMixin._pp_launch_batch
    if getattr(original, "_sglang_fl_ascend_synced", False):
        return

    _require_signature(original, _LAUNCH_PARAMETERS, "_pp_launch_batch")

    @wraps(original)
    def _launch_with_forward_stream_sync(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self.forward_stream.synchronize()
        return result

    _launch_with_forward_stream_sync._sglang_fl_ascend_synced = True
    _launch_with_forward_stream_sync._sglang_fl_original = original
    SchedulerPPMixin._pp_launch_batch = _launch_with_forward_stream_sync
    logger.info("Ascend PP forward-stream sync applied")
