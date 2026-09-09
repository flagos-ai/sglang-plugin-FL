# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backward-compat alias for ``platform_stubs`` (renamed txda -> tsingmicro)."""

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches.platform_stubs import patch

__all__ = ["patch"]
