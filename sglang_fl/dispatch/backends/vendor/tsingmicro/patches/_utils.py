# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for the patches subsystem."""

import torch


def is_txda() -> bool:
    """Check whether the txda platform is available.

    Returns True when ``torch_txda`` is loaded and reports hardware as ready.
    """
    return hasattr(torch, "txda") and torch.txda.is_available()
