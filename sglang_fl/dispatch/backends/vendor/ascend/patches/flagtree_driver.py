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

"""Cache lazy Ascend driver utilities in the affected FlagTree wheel.

FlagTree 0.6.0+ascend3.2 constructs NPUUtils on every ``driver.utils`` access.
Although NPUUtils is a singleton, its non-idempotent __init__ reloads the
extension each time, including on JIT cache hits. Remove this workaround when
that wheel is no longer supported; other versions are deliberately untouched.
"""

from importlib.metadata import PackageNotFoundError, version
import logging
import os
from threading import Lock

logger = logging.getLogger(__name__)


class _CachedDriverUtils:
    """Non-data descriptor: initialize once, then use the instance dictionary."""

    def __init__(self, getter):
        self._getter = getter
        self._reset_lock()
        os.register_at_fork(after_in_child=self._reset_lock)

    def _reset_lock(self):
        self._lock = Lock()

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        # The explicit lock also covers Python 3.12, whose cached_property no
        # longer serializes initialization. Forked workers get a fresh lock.
        with self._lock:
            if "utils" not in instance.__dict__:
                instance.__dict__["utils"] = self._getter(instance)
            return instance.__dict__["utils"]


def patch_flagtree_driver_utils() -> None:
    """Patch the registry class without activating a driver or initializing NPU."""
    try:
        if version("flagtree") != "0.6.0+ascend3.2":
            return
        from triton.backends import backends
    except (PackageNotFoundError, ImportError):
        return

    backend = backends.get("ascend")
    if backend is None:
        return
    driver = backend.driver
    utils = vars(driver).get("utils")
    if (
        driver.__name__ != "NPUDriver"
        or not isinstance(utils, property)
        or utils.fget is None
        or utils.fset is not None
        or utils.fdel is not None
        or getattr(getattr(utils.fget, "__code__", None), "co_names", ())
        != ("NPUUtils",)
    ):
        return

    # Triton loads this class dynamically as ascend.NPUDriver. The ordinary
    # triton.backends.ascend.driver import can refer to a different class.
    driver.utils = _CachedDriverUtils(utils.fget)
    logger.info("Ascend FlagTree driver utils cache patch applied")
