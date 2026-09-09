# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility alias for the TXDA vendor backend.

The TXDA vendor backend was renamed from ``vendor/txda`` to
``vendor/tsingmicro`` during the sglang-plugin-FL refactor.  This package
preserves the legacy ``sglang_fl.dispatch.backends.vendor.txda`` import path
used by run scripts that trigger the monkey-patches *before* any sglang
import (e.g. ``from ...txda.patches.platform_stubs import patch``), because
the plugin's ``load_plugin()`` entry point runs too late for that.

Importing this package applies the same patches as the tsingmicro backend
(including the ``qwen3_asr_config`` AutoConfig ``exist_ok`` patch).
"""

from sglang_fl.dispatch.backends.vendor.tsingmicro.patches import apply_all_txda_patches

apply_all_txda_patches()
