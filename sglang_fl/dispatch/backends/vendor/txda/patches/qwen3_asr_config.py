# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backward-compat alias for ``qwen3_asr_config`` (renamed txda -> tsingmicro)."""

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches.qwen3_asr_config import (
    patch,
    restore,
)

__all__ = ["patch", "restore"]
