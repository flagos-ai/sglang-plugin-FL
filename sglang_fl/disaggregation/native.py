# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""FlagCX data transport with SGLang's native Mooncake control protocol.

Selected by the native region-registration manager API for any model. Bootstrap,
descriptors, state transfer and ACK/error handling stay with the native
manager; only its transport and host-buffer registration are replaced.
"""

import inspect
import os
import threading
from types import SimpleNamespace

from prometheus_client import Counter
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.utils import is_npu
from sglang.srt.utils.network import get_local_ip_auto

from .peer_queues import PeerQueueShards
from .rolling import RollingPDEngine
from .transfer_engine import FlagCXTransferEngine

_LABELS = ["model_name", "role", "tp_rank"]
_BYTES = Counter(
    "sglang:flagcx_total_bytes_transferred", "FlagCX PD write bytes", _LABELS
)
_CALLS = Counter("sglang:flagcx_transfers", "Successful FlagCX PD writes", _LABELS)
_FAILED = Counter(
    "sglang:flagcx_num_failed_transfers", "Failed FlagCX PD writes", _LABELS
)


class _MeteredFlagCXEngine(FlagCXTransferEngine):
    def __init__(self, hostname, gpu_id, ib_device, labels):
        super().__init__(hostname, gpu_id, ib_device)
        self._bytes_counter = _BYTES.labels(**labels)
        self._calls_counter = _CALLS.labels(**labels)
        self._failed_counter = _FAILED.labels(**labels)

    def batch_transfer_sync(self, session_id, buffers, peer_addresses, lengths):
        if not lengths:
            return 0
        result = super().batch_transfer_sync(
            session_id, buffers, peer_addresses, lengths
        )
        if result == 0:
            self._calls_counter.inc()
            self._bytes_counter.inc(sum(lengths))
        else:
            self._failed_counter.inc()
        return result


class FlagcxNativeKVManager(MooncakeKVManager):
    def init_engine(self):
        labels = {
            "model_name": self.server_args.served_model_name,
            "role": self.disaggregation_mode.value,
            "tp_rank": str(self.kv_args.engine_rank),
        }
        ib_device = self.server_args.disaggregation_ib_device
        if ib_device:
            os.environ.setdefault("FLAGCX_IB_HCA", ib_device)
        flagcx = _MeteredFlagCXEngine(
            get_local_ip_auto(),
            self.kv_args.gpu_id,
            ib_device,
            labels,
        )
        self.engine = RollingPDEngine(flagcx)

    def register_buffer_to_engine(self):
        # CUDA auxiliary tensors are host allocations; probing them as CUDA
        # pointers leaves a sticky device error. Ascend's buffers stay on NPU.
        host = set() if is_npu() else set(self.kv_args.aux_data_ptrs or [])
        regions = self._registerable_regions()
        for is_host in (False, True):
            selected = [
                (ptr, size) for ptr, size in regions if (ptr in host) == is_host
            ]
            if selected:
                ptrs, lengths = map(list, zip(*selected))
                register = (
                    self.engine.batch_register_host
                    if is_host
                    else self.engine.batch_register
                )
                if register(ptrs, lengths) != 0:
                    raise RuntimeError("FlagCX PD memory registration failed")


class _BalancedFlagcxNativeKVManager(FlagcxNativeKVManager):
    def __init__(self, *args, **kwargs):
        self._flagcx_peer_queues_lock = threading.Lock()
        self._flagcx_peer_queues = None
        super().__init__(*args, **kwargs)

    def add_transfer_request(
        self,
        bootstrap_room,
        kv_indices,
        index_slice,
        is_last_chunk,
        aux_index=None,
        state_indices=None,
        num_kv_tokens=None,
        trace_ctx=None,
    ):
        from sglang.srt.disaggregation.base.conn import KVPoll
        from sglang.srt.disaggregation.common.utils import TransferKVChunk
        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.observability.trace import TraceNullContext

        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last_chunk or aux_index is not None
        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            return
        if bootstrap_room not in self.transfer_infos:
            return
        selector = self._flagcx_peer_queues
        if selector is None:
            with self._flagcx_peer_queues_lock:
                selector = self._flagcx_peer_queues
                if selector is None:
                    selector = self._flagcx_peer_queues = PeerQueueShards(
                        len(self.transfer_queues)
                    )
        queue_index = selector.select(self.transfer_infos[bootstrap_room])
        self.transfer_queues[queue_index].put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last_chunk=is_last_chunk,
                prefill_aux_index=aux_index,
                state_indices=state_indices,
                num_kv_tokens=num_kv_tokens,
                trace_ctx=TraceNullContext() if trace_ctx is None else trace_ctx,
            )
        )


def _supports_balanced_transfer():
    """Limit the override to the native method contract we can preserve."""
    method = getattr(MooncakeKVManager, "add_transfer_request", None)
    if method is None:
        return False
    try:
        params = list(inspect.signature(method).parameters.values())
    except (TypeError, ValueError):
        return False
    names = (
        "self",
        "bootstrap_room",
        "kv_indices",
        "index_slice",
        "is_last_chunk",
        "aux_index",
        "state_indices",
        "num_kv_tokens",
        "trace_ctx",
    )
    signature_matches = (
        tuple(p.name for p in params) == names
        and all(p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params)
        and all(p.default is inspect.Parameter.empty for p in params[:5])
        and all(p.default is None for p in params[5:])
    )
    if not signature_matches:
        return False

    # Probe the real native method with isolated state, before constructing a
    # manager or starting any transfer. A signature alone does not establish
    # the status, payload and trace semantics needed by the override.
    try:
        from sglang.srt.disaggregation.base.conn import KVPoll
        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.observability.trace import TraceNullContext

        chunks = []
        manager = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.PREFILL,
            request_status={7: object()},
            transfer_infos={7: {"probe:50001": object()}},
            transfer_queues=[SimpleNamespace(put=chunks.append)],
            check_status=lambda _: object(),
        )
        indices, state, trace = object(), object(), object()
        index_slice = slice(1, 3)
        method(manager, 7, indices, index_slice, True, 9, state, 12, trace)
        if len(chunks) != 1:
            return False
        chunk = chunks.pop()
        if not (
            chunk.room == 7
            and chunk.prefill_kv_indices is indices
            and chunk.index_slice == index_slice
            and chunk.is_last_chunk is True
            and chunk.prefill_aux_index == 9
            and chunk.state_indices is state
            and chunk.num_kv_tokens == 12
            and chunk.trace_ctx is trace
        ):
            return False
        method(manager, 7, indices, index_slice, False)
        if len(chunks) != 1 or not isinstance(chunks.pop().trace_ctx, TraceNullContext):
            return False
        manager.check_status = lambda _: KVPoll.Failed
        method(manager, 7, indices, index_slice, False)
        manager.check_status = lambda _: object()
        manager.transfer_infos.clear()
        method(manager, 7, indices, index_slice, False)
        manager.request_status.clear()
        method(manager, 7, indices, index_slice, False)
        if chunks:
            return False
        try:
            method(manager, 7, indices, index_slice, True)
        except AssertionError:
            return True
        return False
    except Exception:  # noqa: BLE001
        # A changed native method remains untouched instead of guessing its
        # semantics. This probe never uses real request or transport state.
        return False


def class_mapping(class_type):
    from sglang.srt.disaggregation.base import KVArgs
    from sglang.srt.disaggregation.mooncake.conn import (
        MooncakeKVBootstrapServer,
        MooncakeKVReceiver,
        MooncakeKVSender,
    )
    from sglang.srt.disaggregation.utils import KVClassType

    if not hasattr(MooncakeKVManager, "_registerable_regions"):
        raise RuntimeError(
            "Native FlagCX PD requires the region-registration manager API"
        )
    if class_type == KVClassType.MANAGER:
        return (
            _BalancedFlagcxNativeKVManager
            if _supports_balanced_transfer()
            else FlagcxNativeKVManager
        )
    return {
        KVClassType.KVARGS: KVArgs,
        KVClassType.SENDER: MooncakeKVSender,
        KVClassType.RECEIVER: MooncakeKVReceiver,
        KVClassType.BOOTSTRAP_SERVER: MooncakeKVBootstrapServer,
    }.get(class_type)
