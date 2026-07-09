"""FlagCX-based OOT communicator for SGLang plugin.

Replaces torch.distributed collective ops with FlagCX API calls,
enabling cross-platform distributed communication on domestic chips.

Adapted from vllm-plugin-FL's PyFlagcxCommunicator pattern:
  https://github.com/flagos-ai/vllm-plugin-FL

Requirements:
  - FLAGCX_PATH env var pointing to FlagCX installation
  - libflagcx.so built at $FLAGCX_PATH/build/lib/
  - FlagCX Python wrapper at $FLAGCX_PATH/plugin/interservice/flagcx_wrapper.py

Usage:
  SGLANG_FL_DIST_BACKEND=flagcx FLAGCX_PATH=/path/to/FlagCX \
      python -m sglang.launch_server --tp 2 ...
"""

import contextlib
import ctypes
import logging
import os
import sys

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _import_flagcx_wrapper():
    """Import FlagCX wrapper, adding FLAGCX_PATH to sys.path if needed."""
    flagcx_path = os.getenv("FLAGCX_PATH")
    if not flagcx_path:
        raise RuntimeError(
            "FLAGCX_PATH environment variable is not set. "
            "Set it to the FlagCX installation directory."
        )
    if not os.path.isdir(flagcx_path):
        raise RuntimeError(f"FLAGCX_PATH={flagcx_path} is not a valid directory.")

    if flagcx_path not in sys.path:
        sys.path.append(flagcx_path)

    from plugin.interservice.flagcx_wrapper import (
        FLAGCXLibrary,
        buffer_type,
        flagcxComm_t,
        flagcxDataTypeEnum,
        flagcxRedOpTypeEnum,
        flagcxUniqueId,
    )

    return (
        FLAGCXLibrary,
        buffer_type,
        flagcxComm_t,
        flagcxDataTypeEnum,
        flagcxRedOpTypeEnum,
        flagcxUniqueId,
    )


class FlagCXCommunicator:
    """OOT communicator using FlagCX as the communication backend.

    Follows the same pattern as vllm-plugin-FL's PyFlagcxCommunicator:
    - Loads libflagcx.so via FLAGCXLibrary
    - Initializes FlagCX communicator via flagcxGetUniqueId + flagcxCommInitRank
    - Overrides all_reduce/reduce_scatter/all_gather to use FlagCX API
    """

    disabled: bool = False

    def __init__(self, group, device):
        self.group = group
        self.device = device

        self.available = False

        self._stream_cache = {}

        # Import FlagCX wrapper
        try:
            (
                FLAGCXLibrary,
                self._buffer_type,
                _,
                self._dtype_enum,
                self._redop_enum,
                flagcxUniqueId,
            ) = _import_flagcx_wrapper()
        except Exception as e:
            logger.warning(
                f"FlagCX wrapper import failed: {e}. Falling back to torch.distributed."
            )
            self.disabled = True
            return

        # Load libflagcx.so
        flagcx_path = os.getenv("FLAGCX_PATH")
        library_path = os.path.join(flagcx_path, "build", "lib", "libflagcx.so")
        try:
            self.flagcx = FLAGCXLibrary(library_path)
        except Exception as e:
            logger.warning(f"Failed to load libflagcx.so from {library_path}: {e}")
            self.disabled = True
            return

        # Get rank/world_size from the process group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)

        if self.world_size <= 1:
            self.disabled = True
            return

        # Initialize FlagCX unique ID (rank 0 generates, then broadcast)
        if self.rank == 0:
            self.unique_id = self.flagcx.flagcxGetUniqueId()
        else:
            self.unique_id = flagcxUniqueId()

        # Broadcast unique ID via torch.distributed (using the CPU group)
        tensor = torch.ByteTensor(list(self.unique_id.internal))
        ranks = dist.get_process_group_ranks(group)
        dist.broadcast(tensor, src=ranks[0], group=group)
        byte_list = tensor.tolist()
        for i, byte_val in enumerate(byte_list):
            self.unique_id.internal[i] = byte_val

        # Ensure device is a torch.device object
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Initialize FlagCX communicator
        device_str = str(self.device)
        if "musa" in device_str:
            device_ctx = torch.musa.device(self.device)
        elif "npu" in device_str:
            device_ctx = contextlib.nullcontext()
        else:
            device_ctx = torch.cuda.device(self.device)
        with device_ctx:
            self.comm = self.flagcx.flagcxCommInitRank(
                self.world_size, self.unique_id, self.rank
            )
            # Warmup: small all_reduce to ensure comm is ready
            warmup_data = torch.zeros(1, device=self.device)
            self._flagcx_all_reduce(warmup_data)
            self._sync_current_stream()
            del warmup_data

        self.available = True
        self.disabled = False
        logger.info(
            f"FlagCX communicator initialized: rank={self.rank}, "
            f"world_size={self.world_size}, device={self.device}"
        )

    def _get_raw_stream_handle(self):
        """Return the vendor-specific raw stream pointer (an int) for FlagCX streamCopy."""
        device_str = str(self.device)
        if "npu" in device_str:
            return torch.npu.current_stream().npu_stream
        if "musa" in device_str:
            return torch.musa.current_stream().musa_stream
        return torch.cuda.current_stream().cuda_stream

    def _sync_current_stream(self):
        """Synchronize the vendor-specific current stream for this device.
        """
        device_str = str(self.device)
        if "npu" in device_str:
            stream = torch.npu.current_stream()
        elif "musa" in device_str:
            stream = torch.musa.current_stream()
        else:
            stream = torch.cuda.current_stream()
        try:
            stream.synchronize()
        except RuntimeError as e:
            logger.error(
                "stream.synchronize() failed on %s: %s", device_str, e
            )
            raise

    def _get_stream(self):
        """Bind a FlagCX stream onto the current vendor stream, with per-handle cache."""
        handle = self._get_raw_stream_handle()
        cached = self._stream_cache.get(handle)
        if cached is not None:
            return cached
        from plugin.interservice.flagcx_wrapper import flagcxStream_t
        new_stream = flagcxStream_t()
        self.flagcx.FLAGCX_CHECK(
            self.flagcx.devHandle.contents.streamCopy(
                ctypes.byref(new_stream),
                ctypes.c_void_p(handle),
            )
        )
        self._stream_cache[handle] = new_stream
        return new_stream

    def _flagcx_all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Internal all_reduce using FlagCX API."""
        out_tensor = torch.empty_like(tensor)
        flagcx_stream = self._get_stream()
        self.flagcx.flagcxAllReduce(
            self._buffer_type(tensor.data_ptr()),
            self._buffer_type(out_tensor.data_ptr()),
            tensor.numel(),
            self._dtype_enum.from_torch(tensor.dtype),
            self._redop_enum.from_torch(torch.distributed.ReduceOp.SUM),
            self.comm,
            flagcx_stream,
        )
        return out_tensor

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """All-reduce using FlagCX. Falls back to torch.distributed if disabled."""
        if self.disabled:
            return super().all_reduce(input_)

        assert input_.device == self.device, (
            f"FlagCX communicator on {self.device}, but tensor on {input_.device}"
        )
        out = self._flagcx_all_reduce(input_)
        # Copy result back to input (in-place semantics expected by SGLang)
        input_.copy_(out)
        return input_

    def reduce_scatter(self, output: torch.Tensor, input_: torch.Tensor):
        """Reduce-scatter using FlagCX."""
        if self.disabled:
            return super().reduce_scatter(output, input_)

        assert input_.device == self.device, (
            f"FlagCX communicator on {self.device}, but tensor on {input_.device}"
        )
        flagcx_stream = self._get_stream()
        self.flagcx.flagcxReduceScatter(
            self._buffer_type(input_.data_ptr()),
            self._buffer_type(output.data_ptr()),
            output.numel(),
            self._dtype_enum.from_torch(input_.dtype),
            self._redop_enum.from_torch(torch.distributed.ReduceOp.SUM),
            self.comm,
            flagcx_stream,
        )

    def all_gather(self, output: torch.Tensor, input_: torch.Tensor):
        """All-gather using FlagCX."""
        if self.disabled:
            return super().all_gather(output, input_)

        assert input_.device == self.device, (
            f"FlagCX communicator on {self.device}, but tensor on {input_.device}"
        )
        flagcx_stream = self._get_stream()
        self.flagcx.flagcxAllGather(
            self._buffer_type(input_.data_ptr()),
            self._buffer_type(output.data_ptr()),
            input_.numel(),
            self._dtype_enum.from_torch(input_.dtype),
            self.comm,
            flagcx_stream,
        )

    def reduce_scatterv(
        self,
        output: torch.Tensor,
        input_: torch.Tensor,
        sizes: list,
        op=None,
    ):
        """Reduce-scatter with variable sizes using grouped Reduce ops."""
        if self.disabled:
            return

        assert input_.device == self.device, (
            f"FlagCX communicator on {self.device}, but tensor on {input_.device}"
        )
        flagcx_stream = self._get_stream()
        self.flagcx.flagcxGroupStart(self.comm)
        split_offset = 0
        for root, split_size in enumerate(sizes):
            chunk = input_[split_offset : split_offset + split_size, ...]
            self.flagcx.flagcxReduce(
                self._buffer_type(chunk.data_ptr()),
                self._buffer_type(output.data_ptr()),
                chunk.numel(),
                self._dtype_enum.from_torch(input_.dtype),
                self._redop_enum.from_torch(torch.distributed.ReduceOp.SUM),
                root,
                self.comm,
                flagcx_stream,
            )
            split_offset += split_size
        self.flagcx.flagcxGroupEnd(self.comm)

    def all_gatherv(
        self,
        output: torch.Tensor,
        input_: torch.Tensor,
        sizes: list,
    ):
        """All-gather with variable sizes using grouped Broadcast ops."""
        if self.disabled:
            return

        assert input_.device == self.device, (
            f"FlagCX communicator on {self.device}, but tensor on {input_.device}"
        )
        assert output.shape[0] == sum(sizes)
        flagcx_stream = self._get_stream()
        self.flagcx.flagcxGroupStart(self.comm)
        split_offset = 0
        for root, split_size in enumerate(sizes):
            dst_slice = output[split_offset : split_offset + split_size]
            self.flagcx.flagcxBroadcast(
                self._buffer_type(input_.data_ptr()),
                self._buffer_type(dst_slice.data_ptr()),
                dst_slice.numel(),
                self._dtype_enum.from_torch(input_.dtype),
                root,
                self.comm,
                flagcx_stream,
            )
            split_offset += split_size
        self.flagcx.flagcxGroupEnd(self.comm)

    def broadcast(self, tensor: torch.Tensor, src: int):
        """Broadcast tensor from src rank."""
        if self.disabled:
            return

        assert tensor.device == self.device, (
            f"FlagCX communicator on {self.device}, but tensor on {tensor.device}"
        )
        flagcx_stream = self._get_stream()
        if src == self.rank:
            sendbuff = self._buffer_type(tensor.data_ptr())
            recvbuff = self._buffer_type(tensor.data_ptr())
        else:
            sendbuff = self._buffer_type()
            recvbuff = self._buffer_type(tensor.data_ptr())
        self.flagcx.flagcxBroadcast(
            sendbuff,
            recvbuff,
            tensor.numel(),
            self._dtype_enum.from_torch(tensor.dtype),
            src,
            self.comm,
            flagcx_stream,
        )

    def send(self, tensor: torch.Tensor, dst: int):
        """Send tensor to destination rank using FlagCX."""
        if self.disabled:
            return

        assert tensor.device == self.device, (
            f"FlagCX communicator on {self.device}, but tensor on {tensor.device}"
        )
        flagcx_stream = self._get_stream()
        self.flagcx.flagcxSend(
            self._buffer_type(tensor.data_ptr()),
            tensor.numel(),
            self._dtype_enum.from_torch(tensor.dtype),
            dst,
            self.comm,
            flagcx_stream,
        )

    def recv(self, tensor: torch.Tensor, src: int):
        """Receive tensor from source rank using FlagCX."""
        if self.disabled:
            return

        assert tensor.device == self.device, (
            f"FlagCX communicator on {self.device}, but tensor on {tensor.device}"
        )
        flagcx_stream = self._get_stream()
        self.flagcx.flagcxRecv(
            self._buffer_type(tensor.data_ptr()),
            tensor.numel(),
            self._dtype_enum.from_torch(tensor.dtype),
            src,
            self.comm,
            flagcx_stream,
        )

    def group_start(self):
        """Start a group of collective operations."""
        self.flagcx.flagcxGroupStart(self.comm)

    def group_end(self):
        """End a group of collective operations."""
        self.flagcx.flagcxGroupEnd(self.comm)

    def __del__(self):
        """Free cached streams and destroy the FlagCX comm on GC / interpreter shutdown."""
        # self.flagcx may be missing (early-disabled path) or unusable (shutdown).
        flagcx = getattr(self, "flagcx", None)
        if flagcx is None:
            return
        cache = getattr(self, "_stream_cache", None)
        if cache:
            for cached_stream in cache.values():
                try:
                    flagcx.adaptor_stream_free(cached_stream)
                except Exception:
                    pass
            cache.clear()
        comm = getattr(self, "comm", None)
        if comm is not None:
            try:
                flagcx.flagcxCommDestroy(comm)
            except Exception:
                pass
            self.comm = None


def create_flagcx_communicator(group, device) -> FlagCXCommunicator:
    """Factory function for FlagCX communicator (registered with GroupCoordinator)."""
    return FlagCXCommunicator(group=group, device=device)
