# Copyright (c) 2026 BAAI. All rights reserved.

"""Shared fixtures for SGLang-FL functional tests."""

import pytest
import torch


def _detect_device_type() -> str:
    """Detect the active torch device type, mirroring ``sglang_fl.platform``.

    Uses the same FlagGems ``DeviceDetector`` the plugin treats as source of
    truth, so NVIDIA -> "cuda", Ascend -> "npu", MUSA -> "musa", ... all resolve
    uniformly. Falls back to "cuda" (legacy behavior) when detection is
    unavailable, so existing CUDA-only environments keep behaving exactly as
    before.
    """
    try:
        try:
            from flag_gems.runtime.backend.device import DeviceDetector
        except ImportError:
            from flag_gems.runtime.backend.device_finder import DeviceDetector
        return DeviceDetector().name
    except Exception:
        return "cuda"


@pytest.fixture(scope="session")
def device():
    """Return the active device for functional operator tests (platform-aware).

    Resolves the device type the same way the plugin does (FlagGems
    DeviceDetector), so each platform gets its native torch device instead of a
    hardcoded CUDA one. Without this, every device-backed functional test
    silently ``pytest.skip``s on non-CUDA platforms, leaving operator-correctness
    coverage at zero while CI reports green.
    """
    device_type = _detect_device_type()
    mod = getattr(torch, device_type, None)
    if mod is None or not callable(getattr(mod, "is_available", None)):
        pytest.skip(f"No torch backend for device type '{device_type}'")
    if not mod.is_available():
        pytest.skip(f"{device_type} is not available")
    return torch.device(device_type)
