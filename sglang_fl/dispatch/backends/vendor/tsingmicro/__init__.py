# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TXDA (TsingMicro) vendor backend for the sglang-plugin-FL dispatch system.

Monkey-patches are now applied via ``patch.py`` (auto-imported by
``sglang_fl._apply_vendor_patches()`` in ``load_plugin()``).  Import this
package directly only when you need the backend class or op registrations.
"""

from sglang_fl.dispatch.backends.vendor.tsingmicro.txda import TxdaBackend  # noqa: F401
