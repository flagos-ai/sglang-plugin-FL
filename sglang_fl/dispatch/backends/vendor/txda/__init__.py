# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TXDA (TsingMicro) vendor backend for the sglang-plugin-FL dispatch system.

Importing this module triggers txda compatibility monkey-patches automatically
(no-op on non-txda platforms).  The patches are idempotent — safe to import
multiple times.
"""

from sglang_fl.dispatch.backends.vendor.txda.patches import apply_all_txda_patches

apply_all_txda_patches()
