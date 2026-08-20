# Copyright (c) 2026 BAAI. All rights reserved.

"""Shared fixtures for SGLang-FL functional tests."""

from typing import Optional

import pytest
import torch


def _detected_device_type() -> Optional[str]:
    try:
        from sglang_fl.platform import _get_device_detector

        detected_name = _get_device_detector().name
        if detected_name:
            return str(detected_name).strip().lower()
    except Exception:
        pass

    for device_type in ("npu", "musa", "cuda"):
        device_module = getattr(torch, device_type, None)
        is_available = getattr(device_module, "is_available", None)
        if callable(is_available) and is_available():
            return device_type

    return None


@pytest.fixture(scope="session")
def device():
    """Return the accelerator selected by the active FlagGems platform."""
    device_type = _detected_device_type()
    if device_type in (None, "cpu"):
        pytest.skip("No supported accelerator is available")

    device_module = getattr(torch, device_type, None)
    if device_module is None:
        pytest.skip(f"torch.{device_type} is not available")

    is_available = getattr(device_module, "is_available", None)
    if callable(is_available) and not is_available():
        pytest.skip(f"torch.{device_type} reports no available accelerator")

    return torch.device(f"{device_type}:0")
