# Copyright (c) 2025 BAAI. All rights reserved.

"""Device helpers for test orchestration.

Adapted from vllm-plugin-FL tests/utils/device_utils.py (#581): the
platform → visibility-env-var mapping used to pin e2e subprocesses to
their scheduler-assigned cards, and vendor-aware device counting (vendor
stacks that attach their own device layer — torch_gcu / torch_musa /
torch_npu — are not visible through torch.cuda).
"""

from __future__ import annotations

import os

# Maps FL_TEST_PLATFORM → the env var that restricts device visibility.
# Each parallel e2e case gets this set to its own device indices so
# concurrent engines stay isolated.
_PLATFORM_VISIBLE_DEVICE_ENV: dict[str, str] = {
    "nvidia": "CUDA_VISIBLE_DEVICES",
    "hygon": "CUDA_VISIBLE_DEVICES",  # DCU stack exposes torch.cuda
    "thead": "CUDA_VISIBLE_DEVICES",  # PPU stack exposes a CUDA-shaped API
    "enflame": "TOPS_VISIBLE_DEVICES",
    "musa": "MUSA_VISIBLE_DEVICES",  # runtime-layer (CUDA_VISIBLE_DEVICES analog); MTHREADS_* is container-layer, read at docker-create time
    "ascend": "ASCEND_RT_VISIBLE_DEVICES",
}


def get_visible_device_env_var() -> str:
    """Return the env var that pins device visibility for this platform."""
    platform = os.environ.get("FL_TEST_PLATFORM", "").lower()
    return _PLATFORM_VISIBLE_DEVICE_ENV.get(platform, "CUDA_VISIBLE_DEVICES")


def get_device_count() -> int:
    """Count available accelerators for the active FL_TEST_PLATFORM.

    Vendor stacks that attach their own device layer (torch_gcu, torch_musa,
    torch_npu) report zero through torch.cuda; probe them directly (same
    probes as the .github/scripts/<platform>/check.sh scripts). Returns 0
    when nothing can be counted — the caller falls back to serial execution.
    """
    platform = os.environ.get("FL_TEST_PLATFORM", "").lower()

    vendor_probes: dict[str, tuple[str, str]] = {
        "enflame": ("torch_gcu", "gcu"),
        "musa": ("torch_musa", "musa"),
        "ascend": ("torch_npu", "npu"),
    }
    if platform in vendor_probes:
        module, attr = vendor_probes[platform]
        try:
            import torch

            __import__(module)
            layer = getattr(torch, attr, None)
            if layer is not None and layer.is_available():
                return layer.device_count()
        except ImportError:
            pass
        return 0

    # CUDA-shaped stacks (nvidia, hygon, thead) and unknown platforms.
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass
    return 0
