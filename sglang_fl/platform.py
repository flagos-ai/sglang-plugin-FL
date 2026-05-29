"""SGLang Platform Plugin — FlagGems-based multi-chip platform.

Registers as an SRTPlatform subclass via the `sglang.srt.platforms` entry_point.
Uses FlagGems DeviceDetector to identify hardware and provides device operations,
memory queries, distributed backend, and subsystem factory methods.

Activation logic:
  - Always activates (returns class path) regardless of hardware.
  - On NVIDIA: is_cuda_alike()=True, uses NCCL by default.
  - On non-NVIDIA (Ascend, etc.): is_cuda_alike()=False, uses hccl/flagcx.

Environment variables:
  SGLANG_FL_DIST_BACKEND=nccl|hccl|flagcx   Override distributed backend
  FLAGCX_PATH=<path>                         If set, default to flagcx backend
"""

import logging
import os
from typing import Optional

import torch

from sglang.srt.platforms.device_mixin import DeviceCapability, PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform

logger = logging.getLogger(__name__)

# Distributed backend mapping: vendor_name -> default backend
_DIST_BACKEND_MAP = {
    "nvidia": "nccl",
    "ascend": "hccl",
    "iluvatar": "nccl",
    "metax": "nccl",
    "cambricon": "cncl",
    "mthreads": "mccl",
}


def _get_device_detector():
    """Lazy import DeviceDetector to avoid import errors when flag_gems not installed."""
    from flag_gems.runtime.backend.device import DeviceDetector

    return DeviceDetector()


class PlatformFL(SRTPlatform):
    """FlagGems-based multi-chip platform for SGLang.

    Provides device abstraction for NVIDIA, Ascend, Iluvatar, MetaX, etc.
    """

    _enum = PlatformEnum.OOT

    def __init__(self):
        super().__init__()
        detector = _get_device_detector()

        # Core device identity from FlagGems
        self._vendor_name: str = detector.vendor_name  # "nvidia", "ascend", ...
        self._device_type: str = detector.name  # "cuda", "npu", ...
        self._dispatch_key: str = detector.dispatch_key  # "CUDA", "NPU", ...
        self._device_count: int = detector.device_count

        # Set class-level attributes expected by DeviceMixin
        self.device_name = self._device_type
        self.device_type = self._device_type

        # torch device module (e.g. torch.cuda, torch.npu)
        self._torch_device_mod = getattr(torch, self._device_type, None)

        # Resolve distributed backend
        self._dist_backend = self._resolve_dist_backend()

        # Set up torch backend device function
        try:
            from flag_gems.runtime import backend

            backend.set_torch_backend_device_fn(self._vendor_name)
        except Exception:
            pass

        logger.info(
            "PlatformFL initialized: vendor=%s, device=%s, dist_backend=%s",
            self._vendor_name,
            self._device_type,
            self._dist_backend,
        )

    def _resolve_dist_backend(self) -> str:
        """Determine distributed backend from env or hardware detection."""
        # Explicit override
        env_backend = os.environ.get("SGLANG_FL_DIST_BACKEND", "").strip()
        if env_backend:
            return env_backend
        # FlagCX if available
        if "FLAGCX_PATH" in os.environ:
            return "flagcx"
        # Default by vendor
        return _DIST_BACKEND_MAP.get(self._vendor_name, "nccl")

    @property
    def vendor_name(self) -> str:
        return self._vendor_name

    # ------------------------------------------------------------------
    # Platform identity overrides
    # ------------------------------------------------------------------

    def is_cuda(self) -> bool:
        return self._device_type == "cuda"

    def is_cuda_alike(self) -> bool:
        """True for devices that expose CUDA-compatible APIs."""
        # Iluvatar uses CUDA API but is not NVIDIA
        if self._vendor_name == "iluvatar":
            return False
        return self._device_type == "cuda"

    def is_out_of_tree(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Active methods (called by SGLang core)
    # ------------------------------------------------------------------

    def get_device_total_memory(self, device_id: int = 0) -> int:
        """Get total device memory in bytes."""
        if self._torch_device_mod is None:
            raise RuntimeError(f"No torch.{self._device_type} module available")
        props = self._torch_device_mod.get_device_properties(device_id)
        return props.total_memory

    def get_current_memory_usage(self, device: Optional[torch.device] = None) -> float:
        """Get current peak memory usage in bytes."""
        if self._torch_device_mod is None:
            return 0.0
        self._torch_device_mod.empty_cache()
        self._torch_device_mod.reset_peak_memory_stats(device)
        return self._torch_device_mod.max_memory_allocated(device)

    # ------------------------------------------------------------------
    # Planned methods (provide implementations for future core migration)
    # ------------------------------------------------------------------

    def get_device(self, local_rank: int) -> torch.device:
        return torch.device(self._device_type, local_rank)

    def set_device(self, device: torch.device) -> None:
        if self._torch_device_mod is not None:
            self._torch_device_mod.set_device(device)

    def get_device_name(self, device_id: int = 0) -> str:
        return self._device_type

    def get_device_capability(self, device_id: int = 0) -> Optional[DeviceCapability]:
        if self._device_type == "cuda":
            major, minor = torch.cuda.get_device_capability(device_id)
            return DeviceCapability(major=major, minor=minor)
        # Non-CUDA devices don't have CUDA compute capability
        return None

    def empty_cache(self) -> None:
        if self._torch_device_mod is not None:
            self._torch_device_mod.empty_cache()

    def synchronize(self) -> None:
        if self._torch_device_mod is not None:
            self._torch_device_mod.synchronize()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        if self._torch_device_mod is None:
            raise RuntimeError(f"No torch.{self._device_type} module available")
        return self._torch_device_mod.mem_get_info(device_id)

    def get_torch_distributed_backend_str(self) -> str:
        return self._dist_backend

    def get_communicator_class(self) -> type | None:
        """Return FlagCX communicator class if flagcx backend is active."""
        if self._dist_backend == "flagcx":
            from sglang_fl.distributed.communicator import CommunicatorFL

            return CommunicatorFL
        return None

    # ------------------------------------------------------------------
    # SRTPlatform subsystem factory methods
    # ------------------------------------------------------------------

    def get_default_attention_backend(self) -> str:
        """Return attention backend name.

        CUDA with FlashAttention available -> "flashinfer" (SGLang default)
        Non-CUDA -> "torch_native" (PyTorch SDPA, registered in attention registry)
        """
        if self._device_type == "cuda":
            return "flashinfer"
        # Non-CUDA: torch_native uses F.scaled_dot_product_attention
        return "torch_native"

    def get_graph_runner_cls(self) -> type:
        """Return graph runner class for this platform."""
        # Import SGLang's default CUDA graph runner
        from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner

        return CudaGraphRunner

    def get_mha_kv_pool_cls(self) -> type:
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        return MHATokenToKVPool

    def get_mla_kv_pool_cls(self) -> type:
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        return MLATokenToKVPool

    def get_nsa_kv_pool_cls(self) -> type:
        from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool

        return NSATokenToKVPool

    def get_paged_allocator_cls(self) -> type:
        from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator

        return PagedTokenToKVPoolAllocator

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    def support_cuda_graph(self) -> bool:
        """CUDA and NPU support graph capture."""
        return self._device_type in ("cuda", "npu")

    def support_piecewise_cuda_graph(self) -> bool:
        return self._device_type == "cuda"

    def is_pin_memory_available(self) -> bool:
        return self._device_type in ("cuda", "npu", "xpu", "musa")

    def supports_fp8(self) -> bool:
        if self._device_type == "cuda":
            cap = self.get_device_capability()
            return cap is not None and cap >= DeviceCapability(8, 9)
        return False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_backend(self) -> None:
        """One-time backend initialization in each worker."""
        logger.info(
            "PlatformFL init_backend: vendor=%s, device=%s",
            self._vendor_name,
            self._device_type,
        )

    # ------------------------------------------------------------------
    # MultiPlatformOp integration
    # ------------------------------------------------------------------

    def get_dispatch_key_name(self) -> str:
        """Return dispatch key for MultiPlatformOp OOT lookup.

        Returns "oot" — our AROUND hook on dispatch_forward() intercepts
        before upstream's single-key lookup and does multi-backend resolution.
        """
        return "oot"

    # ------------------------------------------------------------------
    # Configuration lifecycle
    # ------------------------------------------------------------------

    def apply_server_args_defaults(self, server_args) -> None:
        """Apply platform-specific defaults to server arguments."""
        # Non-CUDA platforms may need attention backend override
        if self._device_type != "cuda":
            if (
                not hasattr(server_args, "attention_backend")
                or server_args.attention_backend is None
            ):
                server_args.attention_backend = "torch_native"
