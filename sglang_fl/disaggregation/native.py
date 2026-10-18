# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""FlagCX data transport with SGLang's native Mooncake control protocol.

Selected by the native region-registration manager API for any model. Bootstrap,
descriptors, state transfer and ACK/error handling stay with the native
manager; only its transport and host-buffer registration are replaced.
"""

import os

from prometheus_client import Counter
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.utils import is_npu
from sglang.srt.utils.network import get_local_ip_auto

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
        labels = dict(
            model_name=self.server_args.served_model_name,
            role=self.disaggregation_mode.value,
            tp_rank=str(self.kv_args.engine_rank),
        )
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
    return {
        KVClassType.KVARGS: KVArgs,
        KVClassType.MANAGER: FlagcxNativeKVManager,
        KVClassType.SENDER: MooncakeKVSender,
        KVClassType.RECEIVER: MooncakeKVReceiver,
        KVClassType.BOOTSTRAP_SERVER: MooncakeKVBootstrapServer,
    }.get(class_type)
