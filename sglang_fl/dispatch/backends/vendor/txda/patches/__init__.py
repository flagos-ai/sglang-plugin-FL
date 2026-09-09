# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backward-compat re-exports for ``vendor/txda/patches``.

Mirrors the modules of ``sglang_fl.dispatch.backends.vendor.tsingmicro.patches``
under the legacy ``txda`` path so existing run scripts keep working without
modification.  All real logic lives in the tsingmicro modules.
"""

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches import (
    apply_all_txda_patches,
    platform_stubs,
    qwen3_asr_config,
)

__all__ = [
    "apply_all_txda_patches",
    "platform_stubs",
    "qwen3_asr_config",
]
