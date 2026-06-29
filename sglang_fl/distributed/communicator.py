"""CommunicatorFL — Full OOT communicator for SGLang plugin.

Wraps FlagCX (when available) or falls back to torch.distributed.
Created per GroupCoordinator via AROUND hook on __init__.

Why AROUND hooks instead of injecting flagcx as torch.distributed backend:
  vLLM lets plugins override platform.dist_backend → passed to init_process_group.
  SGLang SRT picks backend from a hardcoded dict in parallel_state.py (cuda→nccl);
  model_runner.py calls get_default_distributed_backend(device) — a plain dict lookup
  that never consults the platform interface. device_mixin.py defines
  get_torch_distributed_backend_str() as [Planned] and the SRT path does not call it.
  So we hook communication methods at the application layer instead.
"""

import logging
from collections import namedtuple
from typing import List, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])

logger = logging.getLogger(__name__)


class CommunicatorFL:
    """OOT communicator that routes collectives through FlagCX or torch.distributed.

    Lifecycle:
      1. Created by AROUND hook on GroupCoordinator.__init__
      2. Stored as gc.fl_communicator
      3. AROUND hooks on all_reduce/reduce_scatter/etc. delegate to this instance
    """

    disabled: bool = False

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device,
        device_group: ProcessGroup,
        world_size: int,
        rank_in_group: int,
        ranks: List[int],
    ):
        self.cpu_group = cpu_group
        self.device = device
        self.device_group = device_group
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self.ranks = ranks

        # Determine backend
        from sglang_fl.platform import PlatformFL

        try:
            platform = PlatformFL()
            self._dist_backend = platform._dist_backend
        except Exception:
            self._dist_backend = "nccl"

        # Initialize FlagCX communicator if backend is flagcx
        self._flagcx_comm = None
        if self._dist_backend == "flagcx" and world_size > 1:
            try:
                from sglang_fl.distributed.device_communicators.flagcx import (
                    FlagCXCommunicator,
                )

                self._flagcx_comm = FlagCXCommunicator(
                    group=cpu_group,
                    device=device,
                )
                if not self._flagcx_comm.available:
                    logger.warning(
                        "FlagCX communicator init failed, falling back to torch.distributed"
                    )
                    self._flagcx_comm = None
            except Exception as e:
                logger.warning(
                    f"FlagCX communicator creation failed: {e}, using torch.distributed"
                )
                self._flagcx_comm = None

        backend_name = "flagcx" if self._flagcx_comm else "torch.distributed"
        logger.info(
            f"CommunicatorFL created: world_size={world_size}, "
            f"rank={rank_in_group}, backend={backend_name}", flush=True
        )

    # ─── all_reduce ──────────────────────────────────────────────────────────

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """In-place all-reduce. Returns the input tensor (modified in-place)."""
        if self._flagcx_comm and not self._flagcx_comm.disabled:
            out = self._flagcx_comm.all_reduce(input_)
            if out is not None:
                # FlagCX all_reduce returns a new tensor; copy back for in-place semantics
                input_.copy_(out)
                return input_
        # Fallback: torch.distributed
        dist.all_reduce(input_, group=self.device_group)
        return input_

    # ─── reduce_scatter ──────────────────────────────────────────────────────

    def reduce_scatter(self, output: torch.Tensor, input_: torch.Tensor) -> None:
        """Reduce-scatter tensor (in-place into output)."""
        if self._flagcx_comm and not self._flagcx_comm.disabled:
            self._flagcx_comm.reduce_scatter(output, input_)
            return
        dist.reduce_scatter_tensor(output, input_, group=self.device_group)

    # ─── reduce_scatterv ─────────────────────────────────────────────────────

    def reduce_scatterv(
        self,
        input_: torch.Tensor,
        output: Optional[torch.Tensor] = None,
        sizes: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Reduce-scatter with variable sizes per rank."""
        world_size = self.world_size

        if sizes is not None:
            assert len(sizes) == world_size
            assert input_.shape[0] == sum(sizes)
            chunk_size = sizes[self.rank_in_group]
        else:
            assert input_.shape[0] % world_size == 0
            chunk_size = input_.shape[0] // world_size

        output_shape = (chunk_size,) + input_.shape[1:]
        if output is None:
            output = torch.empty(output_shape, dtype=input_.dtype, device=input_.device)
        else:
            assert output.shape == output_shape

        if self._flagcx_comm and not self._flagcx_comm.disabled:
            if sizes is not None and hasattr(self._flagcx_comm, "reduce_scatterv"):
                self._flagcx_comm.reduce_scatterv(output, input_, sizes=sizes)
            else:
                self._flagcx_comm.reduce_scatter(output, input_)
            return output

        # Fallback: torch.distributed (only supports equal sizes)
        dist.reduce_scatter_tensor(output, input_, group=self.device_group)
        return output

    # ─── all_gather ──────────────────────────────────────────────────────────

    def all_gather(self, output: torch.Tensor, input_: torch.Tensor) -> None:
        """All-gather into tensor (in-place into output)."""
        if self._flagcx_comm and not self._flagcx_comm.disabled:
            self._flagcx_comm.all_gather(output, input_)
            return
        dist.all_gather_into_tensor(output, input_, group=self.device_group)

    # ─── all_gatherv ─────────────────────────────────────────────────────────

    def all_gatherv(
        self,
        input_: Union[torch.Tensor, List[torch.Tensor]],
        sizes: Optional[List[int]] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """All-gather with variable sizes per rank."""
        world_size = self.world_size

        def _all_gather_single(inp: torch.Tensor, sizes: Optional[List[int]]):
            input_size = inp.size()
            if sizes is not None:
                assert len(sizes) == world_size
                assert inp.shape[0] == sizes[self.rank_in_group]
                output_size = (sum(sizes),) + input_size[1:]
                # If all sizes equal, treat as uniform
                if all(s == sizes[0] for s in sizes):
                    sizes = None
            else:
                output_size = (input_size[0] * world_size,) + input_size[1:]

            output_tensor = torch.empty(output_size, dtype=inp.dtype, device=inp.device)

            if self._flagcx_comm and not self._flagcx_comm.disabled:
                if sizes is not None and hasattr(self._flagcx_comm, "all_gatherv"):
                    self._flagcx_comm.all_gatherv(output_tensor, inp, sizes=sizes)
                else:
                    self._flagcx_comm.all_gather(output_tensor, inp)
            else:
                dist.all_gather_into_tensor(output_tensor, inp, group=self.device_group)

            return output_tensor

        if isinstance(input_, torch.Tensor):
            input_ = [input_]

        if self._flagcx_comm and not self._flagcx_comm.disabled:
            output_list = []
            self._flagcx_comm.group_start()
            for inp in input_:
                output_list.append(_all_gather_single(inp, sizes=sizes))
            self._flagcx_comm.group_end()
            return output_list
        else:
            output_list = []
            for inp in input_:
                output_list.append(_all_gather_single(inp, sizes=sizes))
            return output_list

    # ─── send ────────────────────────────────────────────────────────────────

    def send(self, tensor: torch.Tensor, dst: int) -> None:
        """Send tensor to destination rank (rank_in_group)."""
        if self._flagcx_comm and not self._flagcx_comm.disabled:
            self._flagcx_comm.send(tensor, dst)
            return
        dist.send(tensor, self.ranks[dst], self.device_group)

    # ─── recv ────────────────────────────────────────────────────────────────

    def recv(self, tensor: torch.Tensor, src: int) -> None:
        """Receive tensor from source rank (rank_in_group)."""
        if self._flagcx_comm and not self._flagcx_comm.disabled:
            self._flagcx_comm.recv(tensor, src)
            return
        dist.recv(tensor, self.ranks[src], self.device_group)

    # ─── broadcast ───────────────────────────────────────────────────────────

    def broadcast(self, input_: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast tensor from src rank (in-place)."""
        if self._flagcx_comm and not self._flagcx_comm.disabled:
            self._flagcx_comm.broadcast(input_, src)
            return input_
        dist.broadcast(input_, src=self.ranks[src], group=self.device_group)
        return input_

    # ─── broadcast_tensor_dict ───────────────────────────────────────────────

    def broadcast_tensor_dict(
        self,
        tensor_dict,
        src: int,
        rank_in_group: int,
        broadcast_object_fn,
    ):
        """Broadcast a tensor dict. GPU tensors go via FlagCX, metadata via CPU.

        Args:
            tensor_dict: Dict to broadcast (only valid on src rank).
            src: Source rank (local rank in group).
            rank_in_group: This rank's position in the group.
            broadcast_object_fn: Callable to broadcast Python objects (CPU group).
        """
        if rank_in_group == src:
            # Sender: split dict into metadata + tensors, broadcast metadata
            metadata_list = []
            tensor_list = []
            for key, value in tensor_dict.items():
                if isinstance(value, torch.Tensor):
                    metadata_list.append(
                        (
                            key,
                            TensorMetadata(
                                value.device.type, value.dtype, value.size()
                            ),
                        )
                    )
                    tensor_list.append(value)
                else:
                    metadata_list.append((key, value))

            broadcast_object_fn(metadata_list, src=src)

            # Broadcast each tensor via FlagCX
            if self._flagcx_comm and not self._flagcx_comm.disabled:
                for tensor in tensor_list:
                    if tensor.numel() == 0:
                        continue
                    if not tensor.is_cpu:
                        self._flagcx_comm.broadcast(tensor, src)
                    else:
                        dist.broadcast(
                            tensor, src=self.ranks[src], group=self.cpu_group
                        )
            else:
                for tensor in tensor_list:
                    if tensor.numel() == 0:
                        continue
                    if tensor.is_cpu:
                        dist.broadcast(
                            tensor, src=self.ranks[src], group=self.cpu_group
                        )
                    else:
                        dist.broadcast(
                            tensor, src=self.ranks[src], group=self.device_group
                        )
            return tensor_dict
        else:
            # Receiver: get metadata, allocate tensors, receive broadcasts
            metadata_list = broadcast_object_fn(None, src=src)
            tensor_dict = {}

            gpu_tensors = []
            for key, value in metadata_list:
                if isinstance(value, TensorMetadata):
                    tensor = torch.empty(
                        value.size, dtype=value.dtype, device=value.device
                    )
                    tensor_dict[key] = tensor
                    if tensor.numel() == 0:
                        continue
                    gpu_tensors.append((tensor, value.device != "cpu"))
                else:
                    tensor_dict[key] = value

            if self._flagcx_comm and not self._flagcx_comm.disabled:
                for tensor, is_gpu in gpu_tensors:
                    if is_gpu:
                        self._flagcx_comm.broadcast(tensor, src)
                    else:
                        dist.broadcast(
                            tensor, src=self.ranks[src], group=self.cpu_group
                        )
            else:
                for tensor, is_gpu in gpu_tensors:
                    if is_gpu:
                        dist.broadcast(
                            tensor, src=self.ranks[src], group=self.device_group
                        )
                    else:
                        dist.broadcast(
                            tensor, src=self.ranks[src], group=self.cpu_group
                        )
            return tensor_dict

    # ─── send_tensor_dict ────────────────────────────────────────────────────

    def send_tensor_dict(
        self,
        tensor_dict,
        dst: int,
        send_object_fn,
        all_gather_group=None,
    ):
        """Send a tensor dict. GPU tensors go via FlagCX, metadata via CPU.

        Args:
            tensor_dict: Dict of tensors/values to send.
            dst: Destination rank (local rank in group).
            send_object_fn: Callable to send Python objects (CPU group).
            all_gather_group: Optional group for send-allgather optimization.
        """
        all_gather_size = 1 if all_gather_group is None else all_gather_group.world_size
        all_gather_rank = (
            0 if all_gather_group is None else all_gather_group.rank_in_group
        )

        metadata_list = []
        tensor_list = []
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor):
                metadata_list.append(
                    (key, TensorMetadata(value.device.type, value.dtype, value.size()))
                )
                tensor_list.append(value)
            else:
                metadata_list.append((key, value))

        # Use async_send=True to avoid deadlock: in PP mode both ranks may
        # call send_tensor_dict before either calls recv_tensor_dict.
        # Synchronous Gloo send would block waiting for a matching recv.
        p2p_works = send_object_fn(metadata_list, dst=dst, async_send=True)

        if self._flagcx_comm and not self._flagcx_comm.disabled:
            for tensor in tensor_list:
                if tensor.numel() == 0:
                    continue
                t = tensor
                if (
                    all_gather_group is not None
                    and tensor.numel() % all_gather_size == 0
                ):
                    t = tensor.reshape(all_gather_size, -1)[all_gather_rank]
                if not t.is_cpu:
                    self._flagcx_comm.send(t, dst)
                else:
                    dist.send(t, self.ranks[dst], self.cpu_group)
        else:
            for tensor in tensor_list:
                if tensor.numel() == 0:
                    continue
                t = tensor
                if (
                    all_gather_group is not None
                    and tensor.numel() % all_gather_size == 0
                ):
                    t = tensor.reshape(all_gather_size, -1)[all_gather_rank]
                if t.is_cpu:
                    dist.send(t, self.ranks[dst], self.cpu_group)
                else:
                    dist.send(t, self.ranks[dst], self.device_group)

        # Return metadata p2p_works so PP mixin can manage their lifecycle.
        # Do NOT wait here — the peer may not have posted recv yet (PP sends
        # before recv, so waiting would deadlock on Gloo isend).
        return p2p_works if p2p_works else []

    # ─── recv_tensor_dict ────────────────────────────────────────────────────

    def recv_tensor_dict(
        self,
        src: int,
        recv_object_fn,
        all_gather_group=None,
    ):
        """Receive a tensor dict. GPU tensors come via FlagCX, metadata via CPU.

        Args:
            src: Source rank (local rank in group).
            recv_object_fn: Callable to receive Python objects (CPU group).
            all_gather_group: Optional group for send-allgather optimization.
        """
        all_gather_size = 1 if all_gather_group is None else all_gather_group.world_size
        all_gather_rank = (
            0 if all_gather_group is None else all_gather_group.rank_in_group
        )

        recv_metadata_list = recv_object_fn(src=src)
        tensor_dict = {}

        recv_ops = []  # (key, tensor, is_gpu, use_all_gather, orig_shape)
        for key, value in recv_metadata_list:
            if isinstance(value, TensorMetadata):
                tensor = torch.empty(value.size, dtype=value.dtype, device=value.device)
                if tensor.numel() == 0:
                    tensor_dict[key] = tensor
                    continue

                use_all_gather = (
                    all_gather_group is not None
                    and tensor.numel() % all_gather_size == 0
                )
                orig_shape = tensor.shape
                if use_all_gather:
                    tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]

                is_gpu = not tensor.is_cpu
                recv_ops.append((key, tensor, is_gpu, use_all_gather, orig_shape))
            else:
                tensor_dict[key] = value

        if self._flagcx_comm and not self._flagcx_comm.disabled:
            for key, tensor, is_gpu, use_all_gather, orig_shape in recv_ops:
                if is_gpu:
                    self._flagcx_comm.recv(tensor, src)
                else:
                    dist.recv(tensor, self.ranks[src], self.cpu_group)
        else:
            for key, tensor, is_gpu, use_all_gather, orig_shape in recv_ops:
                if is_gpu:
                    work = dist.irecv(tensor, self.ranks[src], self.device_group)
                    work.wait()
                else:
                    work = dist.irecv(tensor, self.ranks[src], self.cpu_group)
                    work.wait()

        for key, tensor, is_gpu, use_all_gather, orig_shape in recv_ops:
            if use_all_gather:
                # all_gather on the TP group (not PP group) — use the hooked
                # GroupCoordinator._all_gather_into_tensor via all_gather_group
                full_tensor = torch.empty(
                    orig_shape, dtype=tensor.dtype, device=tensor.device
                )
                all_gather_group._all_gather_into_tensor(full_tensor, tensor)
                tensor = full_tensor
            tensor_dict[key] = tensor

        return tensor_dict
