# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Device info and vendor helpers for sglang_fl.

Kept as a low-level module so both ``platform.py`` and the top-level
``sglang_fl/__init__.py`` can depend on it without circular imports.

``DeviceInfo`` is a live wrapper around FlagGems' ``DeviceDetector`` plus
its ``runtime.backend`` — it aggregates hardware identity + torch device
module + backend accessors in a single place. Constructing it has a side
effect (``set_torch_backend_device_fn``), so callers should go through the
process-wide singleton returned by ``get_device_info()``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

_SUPPORTED_VENDORS: frozenset[str] = frozenset({
    "nvidia", "ascend", "metax", "mthreads",
    "thead", "gcu", "kunlunxin", "hygon",
    "iluvatar", "tsingmicro",
})


class DeviceInfo:
    def __init__(self) -> None:
        try:
            # FlagGems<=5.0.2
            from flag_gems.runtime.backend.device import DeviceDetector
        except (ImportError, FileNotFoundError):
            # FlagGems>5.0.2
            from flag_gems.runtime.backend.device_finder import DeviceDetector

        self.device = DeviceDetector()

        try:
            from flag_gems.runtime import backend

            backend.set_torch_backend_device_fn(self.device.vendor_name)
        except Exception:
            pass

    # ── Identity ──────────────────────────────────────────────────────────

    @property
    def vendor_name(self) -> str:
        """e.g. 'nvidia', 'ascend', 'mthreads'."""
        return self.device.vendor_name

    @property
    def device_type(self) -> str:
        """torch device namespace, e.g. 'cuda', 'npu', 'musa'."""
        return self.device.name

    @property
    def dispatch_key(self) -> str:
        """PyTorch dispatch key, e.g. 'CUDA', 'NPU'."""
        return self.device.dispatch_key

    @property
    def device_count(self) -> int:
        return self.device.device_count

    # ── Torch bindings (via flag_gems backend) ───────────────────────────

    @property
    def torch_device_fn(self):
        from flag_gems.runtime import backend

        return backend.gen_torch_device_object()

    @property
    def torch_backend_device(self):
        from flag_gems.runtime import backend

        return backend.get_torch_backend_device_fn()

    # ── Validation helper ────────────────────────────────────────────────

    def get_supported_device(self) -> bool:
        if self.vendor_name not in _SUPPORTED_VENDORS:
            raise NotImplementedError(f"{self.vendor_name} is not supported now!")
        return True


@lru_cache(maxsize=1)
def get_device_info() -> Optional[DeviceInfo]:
    try:
        return DeviceInfo()
    except Exception as e:
        # Preserve diagnostic detail: caller-side skip logs won't have access
        # to the underlying exception, so surface it once from here.
        import logging

        logging.getLogger(__name__).warning("DeviceDetector unavailable: %s", e)
        return None
