# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Overrides for guarded MThreads implementations, without mutating the env."""

import os


def enabled(name, default=False):
    value = os.getenv(name, "auto").strip().lower()
    return default if value == "auto" else value in ("1", "true", "yes", "on")


def is_s5000(device):
    import torch

    try:
        if isinstance(device, int):
            device = torch.device(f"musa:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        return (
            device.type == "musa"
            and "S5000" in str(torch.musa.get_device_name(device)).upper()
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False
