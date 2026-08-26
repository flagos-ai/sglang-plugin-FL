# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/custom_all_reduce.py

import ctypes
import logging
import os
from contextlib import contextmanager
from functools import partial
from typing import Any, List, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import sglang.srt.distributed.device_communicators.custom_all_reduce_ops as ops
from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    can_use_custom_all_reduce_with_nvlink,
    is_weak_contiguous,
)
from sglang.srt.environ import envs
from sglang_fl.dispatch.backends.vendor.mthreads.jit_custom_ar.fused_rmsnorm import (
    MusaJitCustomAllreduceRMSNorm,
)
from sglang.srt.utils import (
    get_bool_env_var,
    is_cuda,
    is_hip,
    is_musa,
    log_info_on_rank0,
)

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_musa = is_musa()

logger = logging.getLogger(__name__)


def _env_flag(names: tuple[str, ...], default: bool) -> bool:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value.lower() in ("1", "true", "yes", "on")
    return default


def _use_jit_all_reduce() -> bool:
    if _is_cuda:
        return envs.SGLANG_USE_JIT_ALL_REDUCE.get()
    if _is_musa:
        return envs.SGLANG_MUSA_USE_JIT_ALL_REDUCE.get()
    return False


class CustomAllreduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]
    _MAX_CAR_SIZE = 8192 * 1024
    if _is_hip:
        # crossover is at 16MB buffer size for ROCm
        _MAX_CAR_SIZE = 2 * 8192 * 1024
    if _is_musa:
        # XXX (MUSA): 40k prefill can produce ~252MB TP embedding all-reduce
        # inputs. Keep a bounded fast path instead of forcing oversized tensors
        # into the custom kernel without registered buffer space.
        _MAX_CAR_SIZE = 512 * 1024 * 1024

    # max_size: max supported allreduce size
    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_size=_MAX_CAR_SIZE,
    ) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bind to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self._IS_CAPTURING = False
        self.disabled = True  # This can be modified in-place by context manager in piecewise cuda graph runner
        self.original_disabled = True  # To store the original state
        self.use_amd_deterministic_impl = _use_amd_deterministic_impl()

        if not ops.IS_CUSTOM_AR_AVAILABLE:
            # disable because of missing custom allreduce library
            # e.g. in a non-cuda environment
            return

        rank = dist.get_rank(group=group)
        world_size = dist.get_world_size(group=group)

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        full_nvlink = can_use_custom_all_reduce_with_nvlink(
            group=group,
            device=device,
            supported_world_size=self._SUPPORTED_WORLD_SIZES,
            cls_name="CustomAllreduce",
        )
        if full_nvlink is None:
            return  # fail to get nvlink status

        self.group = group
        self.max_size = max_size
        self.rank = rank
        self.world_size = world_size
        self.full_nvlink = full_nvlink

        if not _is_hip:
            # Buffers memory are owned by this Python class and passed to C++.
            # Meta data composes of two parts: meta data for synchronization and a
            # temporary buffer for storing intermediate allreduce results.
            self.meta_ptrs = self.create_shared_buffer(
                ops.meta_size() + max_size, group=group
            )
            # This is a pre-registered IPC buffer. In eager mode, input tensors
            # are first copied into this buffer before allreduce is performed
            self.buffer_ptrs = self.create_shared_buffer(max_size, group=group)
            # This is a buffer for storing the tuples of pointers pointing to
            # IPC buffers from all ranks. Each registered tuple has size of
            # 8*world_size bytes where world_size is at most 8. Allocating 8MB
            # is enough for 131072 such tuples. The largest model I've seen only
            # needs less than 10000 of registered tuples.
            self.rank_data = torch.empty(
                max_size, dtype=torch.uint8, device=self.device
            )
            self._ptr = ops.init_custom_ar(
                self.meta_ptrs, self.rank_data, rank, self.full_nvlink
            )
            ops.register_buffer(self._ptr, self.buffer_ptrs)
        else:
            # meta data buffers need to be "uncached" for signal on MI200
            self.meta = ops.allocate_meta_buffer(ops.meta_size() + max_size)
            self.buffer = torch.empty(max_size, dtype=torch.uint8, device=self.device)
            handle = ops.get_meta_buffer_ipc_handle(self.meta)
            shard_data = (
                bytes(handle),  # ipc handle to base ptr
                0,  # offset of base ptr
            )
            handles, offsets = self._gather_ipc_meta(shard_data)
            self.rank_data = torch.empty(
                max_size, dtype=torch.uint8, device=self.device
            )
            self._ptr = ops.init_custom_ar(
                self.meta, self.rank_data, handles, offsets, rank, self.full_nvlink
            )
            self.register_buffer(self.buffer)

        self.disabled = False
        self.original_disabled = False  # Ensure original_disabled == disabled
        self.tms_cudagraph = envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()

    @staticmethod
    def create_shared_buffer(
        size_in_bytes: int, group: Optional[ProcessGroup] = None
    ) -> List[int]:
        """
        Creates a shared buffer and returns a list of pointers
        representing the buffer on all processes in the group.
        """
        lib = CudaRTLibrary()
        pointer = lib.cudaMalloc(size_in_bytes)
        if _is_musa:
            lib.cudaMemset(pointer, 0, size_in_bytes)
        handle = lib.cudaIpcGetMemHandle(pointer)
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)

        pointers: List[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer.value)  # type: ignore
            else:
                pointers.append(lib.cudaIpcOpenMemHandle(h).value)  # type: ignore

        return pointers

    @staticmethod
    def free_shared_buffer(
        pointers: List[int], group: Optional[ProcessGroup] = None
    ) -> None:
        rank = dist.get_rank(group=group)
        lib = CudaRTLibrary()
        lib.cudaFree(ctypes.c_void_p(pointers[rank]))

    @contextmanager
    def capture(self):
        """
        The main responsibility of this context manager is the
        `register_graph_buffers` call at the end of the context.
        It records all the buffer addresses used in the CUDA graph.
        """
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self.register_graph_buffers()

    def _get_ipc_meta(self, inp: torch.Tensor):
        # _share_cuda_() doesn't accept meta buffer not allocated from
        # PyTorch cache allocator, use direct HIP call to get IPC handle
        handle = ops.get_meta_buffer_ipc_handle(inp)
        shard_data = (
            bytes(handle),  # ipc handle to base ptr
            0,  # offset of base ptr
        )
        return self._gather_ipc_meta(shard_data)

    def _gather_ipc_meta(self, shard_data):
        # Note: don't use `[[None]] * self.world_size` here
        # because it will create a list of the same reference
        all_data: List[Optional[Any]] = [[None] for i in range(self.world_size)]
        all_data[self.rank][0] = shard_data

        ranks = dist.get_process_group_ranks(group=self.group)
        ranks.sort()
        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(
                all_data[i], src=rank, group=self.group, device="cpu"
            )

        # we cannot directly use `dist.all_gather_object` here
        # because it is incompatible with `gloo` backend under inference mode.
        # see https://github.com/pytorch/pytorch/issues/126032 for details.

        handles = []
        offsets = []
        for i in range(len(all_data)):
            handles.append(all_data[i][0][0])  # type: ignore
            offsets.append(all_data[i][0][1])  # type: ignore
        return handles, offsets

    def register_buffer(self, inp: torch.Tensor):
        handles, offsets = self._get_ipc_meta(inp)
        ops.register_buffer(self._ptr, inp, handles, offsets)

    def register_graph_buffers(self):
        if _is_hip:
            handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)
            handles, offsets = self._gather_ipc_meta((bytes(handle), offset))
            log_info_on_rank0(logger, f"Registering {len(offset)} cuda graph addresses")
            ops.register_graph_buffers(self._ptr, handles, offsets)
        else:
            handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)
            log_info_on_rank0(logger, f"Registering {len(offset)} cuda graph addresses")
            # We cannot directly use `dist.all_gather_object` here
            # because it is incompatible with `gloo` backend under inference mode.
            # see https://github.com/pytorch/pytorch/issues/126032 for details.
            all_data = [
                [None, None] for _ in range(dist.get_world_size(group=self.group))
            ]
            all_data[self.rank] = [handle, offset]
            ranks = sorted(dist.get_process_group_ranks(group=self.group))
            for i, rank in enumerate(ranks):
                dist.broadcast_object_list(
                    all_data[i], src=rank, group=self.group, device="cpu"
                )
            # Unpack list of tuples to tuple of lists.
            handles = [d[0] for d in all_data]  # type: ignore
            offsets = [d[1] for d in all_data]  # type: ignore
            ops.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        # for 4 or more non NVLink-capable GPUs, custom allreduce provides
        # little performance improvement over NCCL.
        if not _is_hip:
            if _is_musa:
                # XXX (MUSA): MUSA CAR does not use the CUDA NVLink topology
                # gate; rely on registered-buffer size support instead.
                return inp_size <= self.max_size
            if self.world_size == 2 or self.full_nvlink:
                return inp_size <= self.max_size
            return False

        if _is_hip:
            if self.use_amd_deterministic_impl:
                return True
            if self.full_nvlink:
                return inp_size <= self.max_size
            return False

        return False

    def _all_reduce_impl(self, inp: torch.Tensor, registered: bool):
        out = torch.empty_like(inp)
        if not _is_hip:  # CUDA-like
            if registered:
                ops.all_reduce(self._ptr, inp, out, 0, 0)
            else:
                ops.all_reduce(
                    self._ptr, inp, out, self.buffer_ptrs[self.rank], self.max_size
                )
        elif self.use_amd_deterministic_impl:
            inp_size = inp.numel() * inp.element_size()
            if inp_size < self.max_size:
                reg_buffer = self.buffer.view(inp.dtype)[: inp.numel()]
                ops.deterministic_all_reduce_unreg(self._ptr, inp, reg_buffer, out)
            else:
                self.register_buffer(inp)
                ops.deterministic_all_reduce_reg(self._ptr, inp, out)
        else:  # normal AMD ROCm path
            if registered:
                ops.all_reduce_reg(self._ptr, inp, out)
            else:
                ops.all_reduce_unreg(self._ptr, inp, self.buffer, out)
        return out

    def custom_all_reduce(self, input: torch.Tensor) -> Optional[torch.Tensor]:
        """The main allreduce API that provides support for cuda graph."""
        # When custom allreduce is disabled, this will be None.
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.get_device_module().is_current_stream_capturing():
                return self._all_reduce_impl(input, registered=not self.tms_cudagraph)
            else:
                # Could be warmup OR piecewise cuda graph split op execution.
                # In piecewise cuda graph, split ops run eagerly outside the graph
                # but _IS_CAPTURING is still True. We need to do real all-reduce.
                if is_in_piecewise_cuda_graph():
                    # Split op execution - do real all-reduce
                    return self._all_reduce_impl(input, registered=False)
                else:
                    # True warmup - mimic the allocation pattern since custom
                    # allreduce is out-of-place.
                    return torch.zeros_like(input)
        else:
            return self._all_reduce_impl(input, registered=False)

    def close(self):
        if not self.disabled and self._ptr:
            ops.dispose(self._ptr)
            if _is_cuda:
                self.free_shared_buffer(self.meta_ptrs)
                self.free_shared_buffer(self.buffer_ptrs)
            self._ptr = 0

    def __del__(self):
        self.close()


class MusaJitCustomAllreduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]
    _MAX_CAR_SIZE = 512 * 1024 * 1024
    _RANK_DATA_WIDTH = 8
    _RANK_DATA_ELEMENT_SIZE = ctypes.sizeof(ctypes.c_int64)
    requires_graph_capture_registration_recapture = False

    @classmethod
    def _graph_rank_data_slot_capacity(cls, max_size: int) -> int:
        # Match sgl-kernel's rank-data allocation: max_size bytes are divided
        # into RankData records, each containing eight 64-bit rank pointers.
        return max_size // (cls._RANK_DATA_WIDTH * cls._RANK_DATA_ELEMENT_SIZE)

    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_size=_MAX_CAR_SIZE,
    ) -> None:
        self.disabled = True
        self.original_disabled = True
        self.group = group
        self.max_size = max_size
        self._IS_CAPTURING = False
        self._rank_data_cache: dict[int, torch.Tensor] = {}
        self._rank_data_context_cache: dict[tuple[int, int], int] = {}
        self._unregistered_context_cache: dict[tuple[int, int], int] = {}
        self._opened_ipc_ptrs: dict[bytes, int] = {}
        self._musa_lib = None
        self._mu_pointer_get_attribute = None
        self._last_input_ptr: Optional[int] = None
        self._last_rank_data: Optional[torch.Tensor] = None
        self._graph_inputs: list[tuple[torch.Tensor, Optional[int]]] = []
        self._graph_registered_input_sequence: list[
            tuple[tuple[object, ...], torch.Tensor, torch.Tensor]
        ] = []
        self._graph_registered_sequence_signature: tuple[
            tuple[tuple[object, ...], tuple[int, ...]], ...
        ] = ()
        self._graph_registered_cursor = 0
        self._graph_registered_miss = False
        self._graph_registered_input_enabled = False
        self._shot_decision_cache: dict[tuple[int, bool, bool], int] = {}
        self._context_on_record_graph_input = _env_flag(
            (
                "SGLANG_CUSTOM_AR_CONTEXT_ON_RECORD_GRAPH_INPUT",
                "SGL_CUSTOM_AR_CONTEXT_ON_RECORD_GRAPH_INPUT",
            ),
            False,
        )
        self._launch_context_enabled = _env_flag(
            ("SGLANG_CUSTOM_AR_LAUNCH_CONTEXT", "SGL_CUSTOM_AR_LAUNCH_CONTEXT"),
            True,
        )

        if isinstance(device, int):
            device = torch.device(f"musa:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device

        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        if self.world_size not in self._SUPPORTED_WORLD_SIZES:
            return

        from sglang_fl.dispatch.backends.vendor.mthreads.jit_custom_ar.csrc import (
            allreduce as jit_ar,
        )

        self._jit_ar = jit_ar
        self.meta_ptrs = CustomAllreduce.create_shared_buffer(
            jit_ar.meta_size(self.world_size) + max_size, group=group
        )
        self.buffer_ptrs = CustomAllreduce.create_shared_buffer(max_size, group=group)
        jit_ar.ensure_compiled(self.world_size)
        self._registered_launchers = {}
        self._context_launchers = {}
        self._context_pybind_launchers = {}
        self._context_torchop_launchers = {}
        self._unregistered_launchers = {}
        self._unregistered_pybind_launchers = {}
        self._unregistered_context_pybind_launchers = {}
        self._unregistered_context_pybind_creators = {}
        self._unregistered_context_pybind_disposers = {}
        self._fused_rmsnorm = MusaJitCustomAllreduceRMSNorm(self)
        self._torchop_context_enabled = _env_flag(
            ("SGLANG_CUSTOM_AR_TORCHOP_CONTEXT", "SGL_CUSTOM_AR_TORCHOP_CONTEXT"),
            False,
        )
        self._pybind_context_enabled = _env_flag(
            ("SGLANG_CUSTOM_AR_PYBIND_CONTEXT", "SGL_CUSTOM_AR_PYBIND_CONTEXT"),
            True,
        )
        self._pybind_unregistered_enabled = _env_flag(
            (
                "SGLANG_CUSTOM_AR_PYBIND_UNREGISTERED",
                "SGL_CUSTOM_AR_PYBIND_UNREGISTERED",
            ),
            True,
        )
        self._pybind_unregistered_context_enabled = _env_flag(
            (
                "SGLANG_CUSTOM_AR_PYBIND_UNREGISTERED_CONTEXT",
                "SGL_CUSTOM_AR_PYBIND_UNREGISTERED_CONTEXT",
            ),
            False,
        )
        self._graph_registered_input_enabled = _env_flag(
            (
                "SGLANG_MUSA_CUSTOM_AR_GRAPH_REGISTERED_INPUT",
                "SGL_CUSTOM_AR_GRAPH_REGISTERED_INPUT",
            ),
            # SGLang <=0.5.11 does not have the vendor runtime's graph
            # registration/recapture handshake.  The explicit-input launcher
            # is graph safe and is therefore the compatibility default here.
            False,
        )
        self.requires_graph_capture_registration_recapture = (
            self._graph_registered_input_enabled
        )
        self.rank_data = torch.tensor(
            self.buffer_ptrs + [0] * (8 - self.world_size), dtype=torch.int64
        )
        self.signal_ptrs_cpu = torch.tensor(self.meta_ptrs, dtype=torch.int64)
        # Match sgl-kernel's rank-data capacity. The graph captures a stable
        # RankData* and replay reads the registered pointers from the slot
        # content filled after buffer registration.
        graph_rank_data_slot_capacity = self._graph_rank_data_slot_capacity(max_size)
        if graph_rank_data_slot_capacity == 0:
            raise ValueError(
                "MUSA JIT custom allreduce max_size must fit at least one "
                "rank-data slot (64 bytes)."
            )
        self._graph_rank_data_slots = torch.empty(
            (graph_rank_data_slot_capacity, self._RANK_DATA_WIDTH),
            dtype=torch.int64,
            device=self.device,
        )
        self._graph_rank_data_slot_next = 0
        self._graph_capture_slot_base: Optional[int] = None
        self._graph_capture_slot_count = 0
        self.disabled = False
        self.original_disabled = False

    @contextmanager
    def capture(self):
        try:
            self._graph_inputs.clear()
            self._graph_registered_cursor = 0
            self._graph_registered_miss = False
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self._graph_registered_input_enabled:
                self._graph_inputs.clear()

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False
        if inp.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0 or inp_size > self.max_size:
            return False
        return is_weak_contiguous(inp)

    def _should_fused_rmsnorm_custom_ar(self, inp: torch.Tensor):
        return self._fused_rmsnorm._should_fused_rmsnorm_custom_ar(inp)

    def _get_base_ptr_and_offset(self, inp: torch.Tensor) -> tuple[int, int]:
        ptr_value = int(inp.data_ptr())
        if self._mu_pointer_get_attribute is None:
            self._musa_lib = ctypes.CDLL("libmusa.so")
            self._mu_pointer_get_attribute = self._musa_lib.muPointerGetAttribute
            self._mu_pointer_get_attribute.restype = ctypes.c_int
            self._mu_pointer_get_attribute.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_ulonglong,
            ]
        base_ptr = ctypes.c_void_p()
        err = self._mu_pointer_get_attribute(
            ctypes.byref(base_ptr),
            11,  # MU_POINTER_ATTRIBUTE_RANGE_START_ADDR
            ctypes.c_ulonglong(ptr_value),
        )
        if err != 0:
            raise RuntimeError(f"muPointerGetAttribute failed: {err}")
        base_value = int(base_ptr.value)
        return base_value, ptr_value - base_value

    def _gather_ipc_meta(self, shard_data):
        handle, offset = shard_data
        handle_tensor = torch.tensor(list(handle), dtype=torch.uint8, device="cpu")
        offset_tensor = torch.tensor([offset], dtype=torch.int64, device="cpu")
        handle_list = [torch.empty_like(handle_tensor) for _ in range(self.world_size)]
        offset_list = [torch.empty_like(offset_tensor) for _ in range(self.world_size)]
        dist.all_gather(handle_list, handle_tensor, group=self.group)
        dist.all_gather(offset_list, offset_tensor, group=self.group)
        handles = [bytes(t.tolist()) for t in handle_list]
        offsets = [int(t.item()) for t in offset_list]
        return handles, offsets

    def _local_ipc_record_for_input(self, inp: torch.Tensor) -> tuple[int, bytes, int]:
        ptr_value = int(inp.data_ptr())
        base_value, offset = self._get_base_ptr_and_offset(inp)
        lib = CudaRTLibrary()
        handle = lib.cudaIpcGetMemHandle(ctypes.c_void_p(base_value))
        return ptr_value, bytes(handle), offset

    def _rank_data_from_ipc_records(
        self, ptr_value: int, records: list[tuple[int, bytes, int]]
    ) -> torch.Tensor:
        lib = CudaRTLibrary()
        ptrs: List[int] = []
        for i, (_, h, off) in enumerate(records):
            if i == self.rank:
                ptrs.append(ptr_value)
            else:
                opened_base = self._opened_ipc_ptrs.get(h)
                if opened_base is None:
                    from sglang.srt.distributed.device_communicators.cuda_wrapper import (
                        cudaIpcMemHandle_t,
                    )

                    ipc_handle = cudaIpcMemHandle_t.from_buffer_copy(h)
                    opened_base = lib.cudaIpcOpenMemHandle(ipc_handle).value
                    self._opened_ipc_ptrs[h] = opened_base
                ptrs.append(opened_base + int(off))
        ptrs += [0] * (8 - self.world_size)
        return torch.tensor(ptrs, dtype=torch.int64)

    @staticmethod
    def _rank_data_ptr_tuple(rank_data: torch.Tensor) -> tuple[int, ...]:
        return tuple(int(value) for value in rank_data.tolist())

    def _rank_data_for_input(
        self, inp: torch.Tensor, refresh: bool = False
    ) -> torch.Tensor:
        ptr_value = int(inp.data_ptr())
        if (
            not refresh
            and ptr_value == self._last_input_ptr
            and self._last_rank_data is not None
        ):
            return self._last_rank_data
        if not refresh:
            cached = self._rank_data_cache.get(ptr_value)
            if cached is not None:
                self._last_input_ptr = ptr_value
                self._last_rank_data = cached
                return cached

        _, handle, offset = self._local_ipc_record_for_input(inp)
        handles, offsets = self._gather_ipc_meta((handle, offset))
        rank_data = self._rank_data_from_ipc_records(
            ptr_value,
            [(0, h, off) for h, off in zip(handles, offsets)],
        )
        self._rank_data_cache[ptr_value] = rank_data
        self._last_input_ptr = ptr_value
        self._last_rank_data = rank_data
        return rank_data

    def _context_for_rank_data(
        self, ptr_value: int, shot: int, rank_data: torch.Tensor
    ) -> int:
        key = (ptr_value, int(shot))
        cached = self._rank_data_context_cache.get(key)
        if cached is not None:
            return cached
        context_ptr = self._jit_ar.create_context(
            rank_data,
            self.signal_ptrs_cpu,
            int(self.meta_ptrs[self.rank]),
            self.rank,
            self.world_size,
            shot,
        )
        self._rank_data_context_cache[key] = context_ptr
        return context_ptr

    def _rank_data_for_registered_input(
        self,
        inp: torch.Tensor,
        shot: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        is_graph_launch = (
            self._IS_CAPTURING
            and torch.get_device_module().is_current_stream_capturing()
        )
        if is_graph_launch:
            self._record_graph_input(inp, shot)
            signature = self._graph_input_signature(inp, shot)
            cursor = self._graph_registered_cursor
            self._graph_registered_cursor += 1
            if cursor < len(self._graph_registered_input_sequence):
                cached_signature, rank_data, rank_data_slot = (
                    self._graph_registered_input_sequence[cursor]
                )
                if cached_signature == signature and self._rank_data_ptr_tuple(
                    rank_data
                )[self.rank] == int(inp.data_ptr()):
                    return rank_data_slot
            if self._graph_registered_input_sequence:
                self._graph_registered_miss = True
            return None
        # Registered graph launchers always consume the stable device slot.
        # Eager and SHOT_PUSH paths use their explicit-input launchers instead.
        return None

    @staticmethod
    def _graph_input_signature(
        inp: torch.Tensor, shot: Optional[int]
    ) -> tuple[object, ...]:
        return (
            tuple(int(dim) for dim in inp.shape),
            str(inp.dtype),
            int(inp.numel()),
            int(inp.element_size()),
            shot,
        )

    def _record_graph_input(self, inp: torch.Tensor, shot: Optional[int]) -> None:
        self._graph_inputs.append((inp, int(shot) if shot is not None else None))
        if shot is not None and self._context_on_record_graph_input:
            if self._use_launch_context(shot):
                self._context_for_input(inp, int(shot))

    def _graph_object_broadcast_device(self) -> Union[str, torch.device]:
        try:
            backend = str(dist.get_backend(group=self.group)).lower()
        except Exception:
            backend = ""
        if "gloo" in backend:
            return "cpu"
        return self.device

    def register_graph_buffers(self) -> int:
        if not self._graph_registered_input_enabled:
            self._graph_inputs.clear()
            self._graph_registered_input_sequence.clear()
            self._graph_registered_sequence_signature = ()
            self._graph_registered_cursor = 0
            return 0

        entries = []
        try:
            for inp, shot in tuple(self._graph_inputs):
                ptr_value = int(inp.data_ptr())
                if not self.should_custom_ar(
                    inp
                ) and not self._should_fused_rmsnorm_custom_ar(inp):
                    continue
                signature = self._graph_input_signature(inp, shot)
                ptr_value, handle, offset = self._local_ipc_record_for_input(inp)
                entries.append((ptr_value, handle, offset, shot, signature))

            all_entries = [None for _ in range(self.world_size)]
            all_entries[self.rank] = entries
            ranks = dist.get_process_group_ranks(group=self.group)
            broadcast_device = self._graph_object_broadcast_device()
            for i, rank in enumerate(ranks):
                holder = [all_entries[i]]
                dist.broadcast_object_list(
                    holder, src=rank, group=self.group, device=broadcast_device
                )
                all_entries[i] = holder[0]

            entry_counts = [len(rank_entries or ()) for rank_entries in all_entries]
            if len(set(entry_counts)) != 1:
                raise RuntimeError(
                    "MUSA JIT custom allreduce graph input registration mismatch: "
                    f"rank entry counts are {entry_counts}."
                )

            registered = 0
            new_sequence = []
            new_sequence_signature = []
            slot_base = (
                self._graph_capture_slot_base
                if self._graph_capture_slot_base is not None
                else self._graph_rank_data_slot_next
            )
            for idx, (ptr_value, handle, offset, shot, signature) in enumerate(entries):
                slot_index = slot_base + idx
                if slot_index >= self._graph_rank_data_slots.size(0):
                    raise RuntimeError(
                        "MUSA JIT custom allreduce graph input registration "
                        "exceeds device rank-data slot capacity "
                        f"({self._graph_rank_data_slots.size(0)})."
                    )
                rank_records = []
                rank_signatures = []
                for rank_entries in all_entries:
                    peer_ptr, peer_handle, peer_offset, _, peer_signature = (
                        rank_entries[idx]
                    )
                    rank_records.append((peer_ptr, peer_handle, peer_offset))
                    rank_signatures.append(peer_signature)

                if any(
                    peer_signature != signature for peer_signature in rank_signatures
                ):
                    raise RuntimeError(
                        "MUSA JIT custom allreduce graph input registration "
                        f"order mismatch at entry {idx}: {rank_signatures}."
                    )

                rank_data = self._rank_data_from_ipc_records(ptr_value, rank_records)
                graph_signature = (
                    signature,
                    self._rank_data_ptr_tuple(rank_data),
                )
                rank_data_slot = self._graph_rank_data_slots[slot_index]
                rank_data_slot.copy_(
                    rank_data.to(device=self.device, non_blocking=False)
                )
                new_sequence.append((signature, rank_data, rank_data_slot))
                new_sequence_signature.append(graph_signature)
                self._rank_data_cache[ptr_value] = rank_data
                self._last_input_ptr = ptr_value
                self._last_rank_data = rank_data
            new_sequence_signature = tuple(new_sequence_signature)
            if self._graph_registered_sequence_signature != new_sequence_signature:
                registered += 1
                self._graph_registered_sequence_signature = new_sequence_signature
            elif self._graph_registered_miss:
                registered += 1
            torch.get_device_module().synchronize()
            if self._graph_capture_slot_base is not None:
                self._graph_capture_slot_count = max(
                    self._graph_capture_slot_count, len(new_sequence)
                )
            self._graph_registered_input_sequence = new_sequence
            self._graph_registered_cursor = 0
            self._graph_registered_miss = False
            return registered
        finally:
            self._graph_inputs.clear()

    def begin_graph_capture_registration(self) -> None:
        if not self._graph_registered_input_enabled:
            return
        self._graph_registered_input_sequence.clear()
        self._graph_registered_sequence_signature = ()
        self._graph_registered_cursor = 0
        self._graph_registered_miss = False
        self._graph_capture_slot_base = self._graph_rank_data_slot_next
        self._graph_capture_slot_count = 0

    def end_graph_capture_registration(self) -> None:
        if not self._graph_registered_input_enabled:
            return
        if self._graph_capture_slot_base is not None:
            self._graph_rank_data_slot_next = max(
                self._graph_rank_data_slot_next,
                self._graph_capture_slot_base + self._graph_capture_slot_count,
            )
        self._graph_capture_slot_base = None
        self._graph_capture_slot_count = 0
        self._graph_registered_input_sequence.clear()
        self._graph_registered_sequence_signature = ()
        self._graph_registered_cursor = 0
        self._graph_registered_miss = False

    def prepare_graph_capture(self) -> None:
        if self._graph_registered_input_enabled:
            self._graph_registered_cursor = 0
            self._graph_registered_miss = False

    def prepare_graph_replay(self) -> None:
        if self._graph_registered_input_enabled:
            self._graph_registered_cursor = 0

    def fused_allreduce_rmsnorm(
        self,
        input_: torch.Tensor,
        residual_inp_: torch.Tensor,
        weight_: torch.Tensor,
        eps: float,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        return self._fused_rmsnorm.fused_allreduce_rmsnorm(
            input_, residual_inp_, weight_, eps
        )

    def _launch_registered(
        self, rank_data: torch.Tensor, out: torch.Tensor, shot: int
    ) -> None:
        launcher = self._registered_launchers.get(shot)
        if launcher is None:
            launcher = self._jit_ar.launch_registered_func(self.world_size, shot)
            self._registered_launchers[shot] = launcher
        launcher(
            rank_data,
            self.signal_ptrs_cpu,
            out,
            self.meta_ptrs[self.rank],
            self.rank,
            self.world_size,
            shot,
        )

    def _context_for_input(self, inp: torch.Tensor, shot: int) -> int:
        key = (int(inp.data_ptr()), int(shot))
        cached = self._rank_data_context_cache.get(key)
        if cached is not None:
            return cached
        rank_data = self._rank_data_for_input(inp)
        context_ptr = self._jit_ar.create_context(
            rank_data,
            self.signal_ptrs_cpu,
            self.meta_ptrs[self.rank],
            self.rank,
            self.world_size,
            shot,
        )
        self._rank_data_context_cache[key] = context_ptr
        return context_ptr

    def _launch_registered_context(
        self, context_ptr: int, out: torch.Tensor, shot: int
    ) -> None:
        launcher = self._context_launchers.get(shot)
        if launcher is None:
            launcher = self._jit_ar.launch_context_func(self.world_size, shot)
            self._context_launchers[shot] = launcher
        launcher(int(context_ptr), out, shot)

    def _launch_registered_context_pybind(
        self, context_ptr: int, out: torch.Tensor, shot: int
    ) -> None:
        launcher = self._context_pybind_launchers.get(shot)
        if launcher is None:
            launcher = self._jit_ar.launch_context_pybind_func(self.world_size, shot)
            self._context_pybind_launchers[shot] = launcher
        launcher(int(context_ptr), out, int(shot))

    def _launch_registered_context_torchop(
        self, context_ptr: int, out: torch.Tensor, shot: int
    ) -> None:
        launcher = self._context_torchop_launchers.get(shot)
        if launcher is None:
            launcher = self._jit_ar.launch_context_torchop_func(self.world_size, shot)
            self._context_torchop_launchers[shot] = launcher
        launcher(int(context_ptr), out, int(shot))

    def _use_launch_context(self, shot: int) -> bool:
        if not self._launch_context_enabled:
            return False
        return shot in (
            self._jit_ar.SHOT_TWO_STAGE,
            self._jit_ar.SHOT_TWO_STAGE_512,
        )

    def _use_pybind_context(self, shot: int) -> bool:
        if not self._pybind_context_enabled:
            return False
        return self._use_launch_context(shot)

    def _use_torchop_context(self, shot: int) -> bool:
        if not self._torchop_context_enabled:
            return False
        return self._use_launch_context(shot)

    def _preferred_shot_cached(
        self, input_bytes: int, is_capturing: bool, is_graph_launch: bool
    ) -> int:
        key = (int(input_bytes), bool(is_capturing), bool(is_graph_launch))
        cached = self._shot_decision_cache.get(key)
        if cached is not None:
            return cached
        if (
            is_capturing
            and not self._jit_ar.is_shot_forced()
            and not self._jit_ar.use_push_in_graph()
        ):
            shot = self._jit_ar.preferred_graph_fallback_shot(
                self.world_size, input_bytes
            )
        else:
            shot = self._jit_ar.preferred_shot(self.world_size, input_bytes)
        if (
            shot in (self._jit_ar.SHOT_PUSH, self._jit_ar.SHOT_PUSH_WIDE)
            and is_graph_launch
            and not self._jit_ar.use_push_in_graph()
        ):
            shot = self._jit_ar.preferred_graph_fallback_shot(
                self.world_size, input_bytes
            )
        if (
            shot in (self._jit_ar.SHOT_PUSH, self._jit_ar.SHOT_PUSH_WIDE)
            and self._jit_ar.push_buffer_bytes(input_bytes, self.world_size)
            > self.max_size
        ):
            if self._jit_ar.is_shot_forced():
                raise RuntimeError(
                    "MUSA custom AR push requires "
                    f"{self._jit_ar.push_buffer_bytes(input_bytes, self.world_size)} "
                    f"bytes of staging buffer, but max_size is {self.max_size}"
                )
            shot = self._jit_ar.preferred_fallback_shot(self.world_size, input_bytes)
        self._shot_decision_cache[key] = shot
        return shot

    def _launch_unregistered(
        self, input: torch.Tensor, out: torch.Tensor, shot: int
    ) -> None:
        launcher = self._unregistered_launchers.get(shot)
        if launcher is None:
            launcher = self._jit_ar.launch_unregistered_func(self.world_size, shot)
            self._unregistered_launchers[shot] = launcher
        launcher(
            self.rank_data,
            self.signal_ptrs_cpu,
            input,
            out,
            self.meta_ptrs[self.rank],
            self.buffer_ptrs[self.rank],
            self.max_size,
            self.rank,
            self.world_size,
            shot,
        )

    def _launch_unregistered_pybind(
        self, input: torch.Tensor, out: torch.Tensor, shot: int
    ) -> None:
        launcher = self._unregistered_pybind_launchers.get(shot)
        if launcher is None:
            launcher = self._jit_ar.launch_unregistered_pybind_func(
                self.world_size, shot
            )
            self._unregistered_pybind_launchers[shot] = launcher
        launcher(
            self.rank_data,
            self.signal_ptrs_cpu,
            input,
            out,
            int(self.meta_ptrs[self.rank]),
            int(self.buffer_ptrs[self.rank]),
            int(self.rank),
            int(self.world_size),
            int(shot),
        )

    def _unregistered_context_for_input(self, input: torch.Tensor, shot: int) -> int:
        key = (int(input.data_ptr()), int(shot))
        cached = self._unregistered_context_cache.get(key)
        if cached is not None:
            return cached
        creator = self._unregistered_context_pybind_creators.get(shot)
        if creator is None:
            creator = self._jit_ar.create_unregistered_context_pybind_func(
                self.world_size, shot
            )
            self._unregistered_context_pybind_creators[shot] = creator
        context_ptr = int(
            creator(
                self.rank_data,
                self.signal_ptrs_cpu,
                input,
                int(self.meta_ptrs[self.rank]),
                int(self.buffer_ptrs[self.rank]),
                int(self.rank),
                int(self.world_size),
            )
        )
        self._unregistered_context_cache[key] = context_ptr
        return context_ptr

    def _launch_unregistered_context_pybind(
        self, context_ptr: int, out: torch.Tensor, shot: int
    ) -> None:
        launcher = self._unregistered_context_pybind_launchers.get(shot)
        if launcher is None:
            launcher = self._jit_ar.launch_unregistered_context_pybind_func(
                self.world_size, shot
            )
            self._unregistered_context_pybind_launchers[shot] = launcher
        launcher(int(context_ptr), out, int(shot))

    def custom_all_reduce(self, input: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.should_custom_ar(input):
            return None
        input_bytes = input.numel() * input.element_size()
        is_graph_launch = (
            self._IS_CAPTURING
            and torch.get_device_module().is_current_stream_capturing()
        )
        shot = self._preferred_shot_cached(
            input_bytes, self._IS_CAPTURING, is_graph_launch
        )

        if shot in (self._jit_ar.SHOT_PUSH, self._jit_ar.SHOT_PUSH_WIDE):
            out = torch.empty_like(input)
            if not self._IS_CAPTURING and self._pybind_unregistered_enabled:
                if self._pybind_unregistered_context_enabled:
                    context_ptr = self._unregistered_context_for_input(input, shot)
                    self._launch_unregistered_context_pybind(context_ptr, out, shot)
                else:
                    self._launch_unregistered_pybind(input, out, shot)
            else:
                self._launch_unregistered(input, out, shot)
            return out

        out = None
        if self._IS_CAPTURING:
            out = torch.empty_like(input)
            if is_graph_launch:
                if self._graph_registered_input_enabled:
                    rank_data = self._rank_data_for_registered_input(input, shot)
                    if rank_data is not None:
                        self._launch_registered(rank_data, out, shot)
                    else:
                        # A registration miss must not silently turn the
                        # all-reduce into zeros.  Keep correctness during a
                        # recapture mismatch by using the explicit-input
                        # launcher; registered replay remains the fast path.
                        self._launch_unregistered(input, out, shot)
                else:
                    self._launch_unregistered(input, out, shot)
            else:
                if is_in_piecewise_cuda_graph():
                    self._launch_unregistered(input, out, shot)
                else:
                    # Run the real eager-style path during graph warmup so the
                    # JIT module is built before stream capture. The first
                    # actual capture can still record a placeholder while IPC
                    # rank data is registered for the recapture.
                    self._launch_unregistered(input, out, shot)
        else:
            out = torch.empty_like(input)
            # Match the sgl custom AR eager path: stage through the shared
            # buffer instead of launching from an opaque registered input
            # pointer. The unregistered launch takes `input` as an explicit
            # tensor argument and orders a same-stream D2D copy before AR,
            # preserving producer/lifetime dependencies under serving buffer
            # reuse. Graph capture uses registered inputs by default and can
            # disable them via SGLANG_MUSA_CUSTOM_AR_GRAPH_REGISTERED_INPUT=0.
            if self._pybind_unregistered_enabled:
                self._launch_unregistered_pybind(input, out, shot)
            else:
                self._launch_unregistered(input, out, shot)
        return out

    def close(self):
        if not self.disabled and dist.is_initialized():
            for (_, shot), context_ptr in tuple(self._rank_data_context_cache.items()):
                self._jit_ar.dispose_context(context_ptr, self.world_size, shot)
            self._rank_data_context_cache.clear()
            for (_, shot), context_ptr in tuple(
                self._unregistered_context_cache.items()
            ):
                disposer = self._unregistered_context_pybind_disposers.get(shot)
                if disposer is None:
                    disposer = self._jit_ar.dispose_unregistered_context_pybind_func(
                        self.world_size, shot
                    )
                    self._unregistered_context_pybind_disposers[shot] = disposer
                disposer(int(context_ptr))
            self._unregistered_context_cache.clear()
            lib = CudaRTLibrary()
            for ptr in self._opened_ipc_ptrs.values():
                lib.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr))
            self._opened_ipc_ptrs.clear()
            self._rank_data_cache.clear()
            self._graph_registered_input_sequence.clear()
            self._graph_registered_sequence_signature = ()
            self._graph_registered_cursor = 0
            self._last_input_ptr = None
            self._last_rank_data = None
            CustomAllreduce.free_shared_buffer(self.buffer_ptrs, group=self.group)
            CustomAllreduce.free_shared_buffer(self.meta_ptrs, group=self.group)
        self.disabled = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def dispatch_custom_allreduce():
    """Return the CustomAllreduce class to use (aiter on ROCm if enabled).

    On AMD with 1-stage AR enabled, use sglang's CustomAllreduce.
    Otherwise use AiterCustomAllreduce if available.

    Set SGLANG_USE_JIT_ALL_REDUCE=1 for CUDA or
    SGLANG_MUSA_USE_JIT_ALL_REDUCE=1 for MUSA to use the JIT-compiled implementation.
    """
    if _use_jit_all_reduce():
        if _is_cuda:
            from .custom_all_reduce_v2 import CustomAllReduceV2

            logger.debug("[AR] Using CustomAllReduceV2 (JIT-compiled)")
            return CustomAllReduceV2
        if _is_musa:
            logger.debug("[AR] Using MusaJitCustomAllreduce (JIT-compiled)")
            return MusaJitCustomAllreduce

    if _is_cuda:
        return CustomAllreduce

    assert _is_hip

    if envs.SGLANG_USE_1STAGE_ALLREDUCE.is_set():
        if envs.SGLANG_USE_1STAGE_ALLREDUCE.get():
            logger.debug(
                "[AR] All-reduce: 1-stage kernel (SGLANG_USE_1STAGE_ALLREDUCE=1)"
            )
        else:
            logger.debug("[AR] All-reduce: default (SGLANG_USE_1STAGE_ALLREDUCE=0)")
    elif envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get():
        logger.debug(
            "[AR] All-reduce: 1-stage kernel (deterministic inference enabled)"
        )
    else:
        logger.debug("[AR] All-reduce: default")

    # On AMD with 1-stage AR, use sglang's CustomAllreduce
    # (AiterCustomAllreduce doesn't have deterministic_all_reduce method)
    if _use_amd_deterministic_impl():
        return CustomAllreduce

    if get_bool_env_var("SGLANG_USE_AITER_AR", default="true"):
        try:
            from aiter.dist.device_communicators.custom_all_reduce import (
                CustomAllreduce as AiterCustomAllreduce,
            )

            logger.info("[AR] Using AiterCustomAllreduce (AMD default)")
            tms_cudagraph = envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()
            return partial(
                AiterCustomAllreduce,
                enable_register_for_capturing=not tms_cudagraph,
            )
        except ImportError as e:
            logger.warning(
                "[AR] Aiter custom all-reduce not available; "
                "falling back to sglang CustomAllreduce. Details: %s",
                e,
            )
            return CustomAllreduce

    return CustomAllreduce


def _use_amd_deterministic_impl() -> bool:
    if not _is_hip:  # CUDA is always deterministic
        return False
    if envs.SGLANG_USE_1STAGE_ALLREDUCE.is_set():
        return envs.SGLANG_USE_1STAGE_ALLREDUCE.get()
    else:
        return envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
