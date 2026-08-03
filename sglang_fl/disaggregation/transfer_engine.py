# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import threading
import torch

from typing import List, Optional
from sglang.srt.platforms import current_platform
from sglang.srt.utils.network import NetworkAddress

logger = logging.getLogger(__name__)

# Module-level shared engine instance, set by init_flagcx_transfer_engine().
_flagcx_transfer_engine: Optional["FlagCXTransferEngine"] = None


class FlagCXTransferEngine:
    """Shared FlagCX transfer engine for one-sided RDMA KV transfer.

    Addresses a remote peer by a ``"host:port"`` session string and performs
    synchronous batched one-sided writes into the peer's registered memory.
    """

    def __init__(
        self,
        hostname: str,
        gpu_id: Optional[int] = None,
        ib_device: Optional[str] = None,
    ):
        # Reuse the collective communicator's FLAGCX_PATH/sys.path handling so
        # there is a single place that knows how to locate the FlagCX wrapper.
        from sglang_fl.distributed.device_communicators.flagcx import (
            _import_flagcx_wrapper,
        )

        flagcx_library = _import_flagcx_wrapper()[0]

        self.hostname = hostname
        self.gpu_id = gpu_id if gpu_id is not None else 0
        self.ib_device = ib_device
        current_platform.set_device(current_platform.get_device(self.gpu_id))
        self.flagcx = flagcx_library()
        self.engine = self.flagcx.flagcxP2pEngineCreate()
        self.rpc_port = self.flagcx.flagcxP2pGetRpcPort(self.engine)
        self.flagcx.flagcxP2pStartRpcServer(self.engine)
        self.session_id = NetworkAddress(
            self.hostname, self.rpc_port
        ).to_host_port_str()

        # Cache of session_id -> P2P connection handle. The first GetConn for a
        # session performs the QP + descriptor-table handshake, so cache it.
        self.session_conns: dict = {}
        self.conn_lock = threading.Lock()

        logger.debug(
            "FlagCX transfer engine initialized: session_id=%s gpu_id=%s ib_device=%s",
            self.session_id,
            self.gpu_id,
            self.ib_device,
        )

    def register(self, ptr, length):
        """Register a single memory region for one-sided access."""
        try:
            self.flagcx.flagcxP2pRegister(self.engine, ptr, length)
        except Exception:
            logger.debug("FlagCX memory registration %s failed.", ptr)

    def register_host(self, ptr, length):
        """Register a single host (CPU) memory region for one-sided access.

        Uses the host-registration path so FlagCX skips the CUDA IPC probe
        (CPU pointers can't yield a CUDA IPC handle; probing them leaves a
        sticky CUDA error that would poison the next torch call).
        """
        self.flagcx.flagcxP2pRegisterHost(self.engine, ptr, length)

    def deregister(self, ptr):
        """FlagCX has no explicit deregister; memory is released on engine
        destroy."""
        logger.debug("FlagCX deregister is a no-op for %s.", ptr)

    def batch_register(self, ptrs: List[int], lengths: List[int]) -> int:
        """Batch register multiple memory regions."""
        try:
            for ptr, length in zip(ptrs, lengths):
                self.flagcx.flagcxP2pRegister(self.engine, ptr, length)
        except Exception:
            logger.debug("FlagCX batch memory registration failed.")
            return -1
        return 0

    def batch_register_host(self, ptrs: List[int], lengths: List[int]) -> int:
        """Batch register host (CPU) memory regions, skipping the CUDA IPC
        probe entirely (see register_host)."""
        for ptr, length in zip(ptrs, lengths):
            self.flagcx.flagcxP2pRegisterHost(self.engine, ptr, length)
        return 0

    def batch_deregister(self, ptrs: List[int]) -> int:
        """Batch deregister; no-op for FlagCX."""
        return 0

    def _get_conn(self, session_id: str):
        conn = self.session_conns.get(session_id)
        if conn is not None:
            return conn
        with self.conn_lock:
            conn = self.session_conns.get(session_id)
            if conn is None:
                conn = self.flagcx.flagcxP2pGetConn(self.engine, session_id)
                self.session_conns[session_id] = conn
            return conn

    def transfer_sync(
        self, session_id: str, buffer: int, peer_buffer_address: int, length: int
    ) -> int:
        """Synchronously write data to the specified remote address."""
        return self.batch_transfer_sync(
            session_id, [buffer], [peer_buffer_address], [length]
        )

    def batch_transfer_sync(
        self,
        session_id: str,
        buffers: List[int],
        peer_buffer_addresses: List[int],
        lengths: List[int],
    ) -> int:
        """Synchronously write data to the specified remote addresses in
        batches. Returns 0 on success, -1 on failure."""
        try:
            conn = self._get_conn(session_id)
            self.flagcx.flagcxP2pBatchWriteSync(
                conn, buffers, peer_buffer_addresses, lengths
            )
        except Exception:
            logger.debug(
                "Failed to batch transfer data. Buffers: %s, Session: %s, "
                "Peer addresses: %s",
                buffers,
                session_id,
                peer_buffer_addresses,
            )
            return -1
        return 0

    def get_session_id(self):
        return self.session_id

    def get_ib_device(self):
        return self.ib_device


def init_flagcx_transfer_engine(
    hostname: str,
    gpu_id: Optional[int] = None,
    ib_device: Optional[str] = None,
) -> FlagCXTransferEngine:
    """
    Initialize the shared FlagCXTransferEngine. Note: if already initialized,
    returns the existing instance. Called lazily from FlagcxKVManager when the
    FlagCX transfer backend is selected for PD disaggregation.
    """
    global _flagcx_transfer_engine
    if _flagcx_transfer_engine is not None:
        return _flagcx_transfer_engine
    _flagcx_transfer_engine = FlagCXTransferEngine(
        hostname=hostname, gpu_id=gpu_id, ib_device=ib_device
    )
    return _flagcx_transfer_engine


def get_flagcx_transfer_engine() -> Optional[FlagCXTransferEngine]:
    """Return the shared FlagCXTransferEngine if initialized, else None."""
    return _flagcx_transfer_engine
