# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
# Adapted from the native SGLang MusaJitCustomAllreduce IPC lifecycle.
"""TP8 MUSA communication buffers owned by SGLang's GroupCoordinator."""

import ctypes
from contextlib import contextmanager

import torch
import torch.distributed as dist
from sglang.srt.distributed.device_communicators.cuda_wrapper import (
    CudaRTLibrary,
    cudaIpcMemHandle_t,
)


class MusaCustomAllreduce:
    disabled = False
    original_disabled = False
    requires_graph_capture_registration_recapture = False

    def __init__(self, group, device, max_size=2 * 1024 * 1024):
        from flaggems_sglang.runtime.backend._mthreads.communication import all_reduce

        self.kernels = all_reduce
        self.group, self.device = group, torch.device(device)
        self.rank, self.world_size = dist.get_rank(group), dist.get_world_size(group)
        if self.world_size != 8 or self.device.type != "musa":
            raise ValueError("MUSA custom AllReduce requires a local TP8 group")
        self.max_size = max_size
        self.library = CudaRTLibrary()
        self._driver = ctypes.CDLL("libmusa.so")
        self._range_start = self._driver.muPointerGetAttribute
        self._range_start.restype = ctypes.c_int
        self._range_start.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ulonglong]
        self._ipc_close = self.library.lib.musaIpcCloseMemHandle
        self._ipc_close.argtypes = [ctypes.c_void_p]
        self._ipc_close.restype = ctypes.c_int
        signal_bytes = self.kernels.prepare()
        self.meta_ptrs = self._allocate_shared(signal_bytes + max_size)
        self.buffer_ptrs = self._allocate_shared(max_size)
        self.signal_ptrs = torch.tensor(self.meta_ptrs, dtype=torch.int64, device="cpu")
        self.rank_data = torch.tensor(self.buffer_ptrs, dtype=torch.int64, device="cpu")
        self.graph_rank_data = torch.empty((8192, 8), dtype=torch.int64, device=device)
        self._graph_next = 0
        self._graph_inputs = []
        self._opened = {}
        self._capturing = False

    def _allocate_shared(self, size):
        pointer = self.library.cudaMalloc(size)
        self.library.cudaMemset(pointer, 0, size)
        handles = [None] * self.world_size
        dist.all_gather_object(
            handles, self.library.cudaIpcGetMemHandle(pointer), group=self.group
        )
        return [
            pointer.value
            if rank == self.rank
            else self.library.cudaIpcOpenMemHandle(handle).value
            for rank, handle in enumerate(handles)
        ]

    def should_custom_ar(self, tensor):
        # Only the measured batch-64 decode graphs select this backend. Eager
        # execution and smaller graphs retain the established FlagCX sum.
        return (
            not self.disabled
            and self._capturing
            and tensor.device == self.device
            and tensor.dtype == torch.bfloat16
            and tensor.is_contiguous()
            and tensor.ndim == 2
            and tensor.shape[0] == 64
            and tensor.shape[1] in (2560, 10240)
            and tensor.numel() * tensor.element_size() <= self.max_size
            and tensor.data_ptr() % 16 == 0
        )

    @contextmanager
    def capture(self):
        first = len(self._graph_inputs)
        self._capturing = True
        try:
            yield
        finally:
            self._capturing = False
        self._register_graph_inputs(first)

    def _register_graph_inputs(self, first):
        entries = []
        for tensor in self._graph_inputs[first:]:
            base = ctypes.c_void_p()
            status = self._range_start(ctypes.byref(base), 11, tensor.data_ptr())
            if status != 0:
                raise RuntimeError(f"muPointerGetAttribute failed: {status}")
            handle = bytes(self.library.cudaIpcGetMemHandle(base))
            entries.append((tensor.data_ptr(), handle, tensor.data_ptr() - base.value))
        peers = [None] * self.world_size
        dist.all_gather_object(peers, entries, group=self.group)
        if any(len(peer) != len(entries) for peer in peers):
            raise RuntimeError("MUSA graph collective counts differ across ranks")
        for index in range(len(entries)):
            pointers = []
            for rank, peer in enumerate(peers):
                pointer, handle, offset = peer[index]
                if rank != self.rank:
                    if handle not in self._opened:
                        self._opened[handle] = self.library.cudaIpcOpenMemHandle(
                            cudaIpcMemHandle_t.from_buffer_copy(handle)
                        ).value
                    pointer = self._opened[handle] + offset
                pointers.append(pointer)
            self.graph_rank_data[first + index].copy_(
                torch.tensor(pointers, dtype=torch.int64, device="cpu")
            )
        torch.musa.synchronize()
        dist.barrier(group=self.group)

    def custom_all_reduce(self, tensor):
        size = tensor.numel() * tensor.element_size()
        shot = 4 if self._capturing and size <= 256 * 1024 else 2
        out = torch.empty_like(tensor)
        if self._capturing and torch.musa.is_current_stream_capturing():
            if self._graph_next == self.graph_rank_data.shape[0]:
                raise RuntimeError("MUSA graph collective buffer capacity exceeded")
            slot = self.graph_rank_data[self._graph_next]
            self._graph_next += 1
            self._graph_inputs.append(tensor)
            self.kernels.module(shot).sgl_musa_custom_ar_launch(
                slot,
                self.signal_ptrs,
                out,
                self.meta_ptrs[self.rank],
                self.rank,
                8,
                shot,
            )
        else:
            self.kernels.pybind_module(shot).launch_unregistered(
                self.rank_data,
                self.signal_ptrs,
                tensor,
                out,
                self.meta_ptrs[self.rank],
                self.buffer_ptrs[self.rank],
                self.rank,
                8,
                shot,
            )
        return out

    def reset_graph_buffers(self):
        """Called only after the framework retires the previous decode graphs."""
        torch.musa.synchronize()
        dist.barrier(group=self.group)
        self._graph_inputs.clear()
        self._graph_next = 0
        for pointer in self._opened.values():
            self.library.CUDART_CHECK(self._ipc_close(ctypes.c_void_p(pointer)))
        self._opened.clear()

    def close(self):
        if self.disabled:
            return
        self.reset_graph_buffers()
        for pointers in (self.buffer_ptrs, self.meta_ptrs):
            for rank, pointer in enumerate(pointers):
                if rank != self.rank:
                    self.library.CUDART_CHECK(self._ipc_close(ctypes.c_void_p(pointer)))
            self.library.cudaFree(ctypes.c_void_p(pointers[self.rank]))
        self.disabled = True
