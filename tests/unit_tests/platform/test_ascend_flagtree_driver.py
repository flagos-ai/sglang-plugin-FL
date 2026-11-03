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

from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from importlib.metadata import PackageNotFoundError
import os
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

from sglang_fl.dispatch.backends.vendor.ascend.patches import flagtree_driver as patch


@pytest.fixture
def registry(monkeypatch):
    calls = []

    def new_utils():
        calls.append(object())
        time.sleep(0.005)
        return calls[-1]

    monkeypatch.setitem(globals(), "NPUUtils", new_utils)

    class NPUDriver:
        @property
        def utils(self):
            return NPUUtils()  # noqa: F821 - provided by the fixture

    # Deliberately different from the dynamically loaded registry class.
    alias = type("NPUDriver", (), {"utils": NPUDriver.utils})
    module = ModuleType("triton.backends")
    module.backends = {"ascend": SimpleNamespace(driver=NPUDriver)}
    monkeypatch.setitem(sys.modules, "triton.backends", module)
    monkeypatch.setattr(patch, "version", lambda name: "0.6.0+ascend3.2")
    return SimpleNamespace(driver=NPUDriver, alias=alias, calls=calls, module=module)


def test_lazy_registry_patch_is_idempotent(registry):
    original = registry.alias.utils
    patch.patch_flagtree_driver_utils()
    descriptor = registry.driver.utils
    patch.patch_flagtree_driver_utils()
    assert registry.driver.utils is descriptor
    assert registry.alias.utils is original
    assert registry.calls == []
    driver = registry.driver()
    assert driver.utils is driver.utils
    assert len(registry.calls) == 1


def test_concurrent_first_access_initializes_once(registry):
    patch.patch_flagtree_driver_utils()
    driver = registry.driver()
    with ThreadPoolExecutor(max_workers=16) as executor:
        values = list(executor.map(lambda _: driver.utils, range(32)))
    assert len(registry.calls) == 1
    assert all(value is values[0] for value in values)


@pytest.mark.parametrize("version", [None, "0.6.0", "0.7.0+ascend3.2"])
def test_unaffected_or_missing_distribution_is_untouched(
    registry, monkeypatch, version
):
    original = registry.driver.utils

    def get_version(name):
        if version is None:
            raise PackageNotFoundError(name)
        return version

    monkeypatch.setattr(patch, "version", get_version)
    patch.patch_flagtree_driver_utils()
    assert registry.driver.utils is original
    assert registry.calls == []


@pytest.mark.parametrize(
    "layout", ["missing", "eager", "cached", "different", "setter"]
)
def test_other_driver_layouts_are_untouched(registry, layout):
    if layout == "missing":
        registry.module.backends.clear()
    elif layout == "eager":
        del registry.driver.utils
    elif layout == "cached":
        descriptor = cached_property(registry.driver.utils.fget)
        descriptor.__set_name__(registry.driver, "utils")
        registry.driver.utils = descriptor
    elif layout == "different":
        registry.driver.utils = property(lambda self: object())
    else:
        registry.driver.utils = registry.driver.utils.setter(lambda self, value: None)
    original = vars(registry.driver).get("utils")
    patch.patch_flagtree_driver_utils()
    assert vars(registry.driver).get("utils") is original
    assert registry.calls == []


def test_initialization_failure_can_retry(registry):
    original = registry.driver.utils.fget
    attempts = []

    def getter(self):
        attempts.append(None)
        if len(attempts) == 1:
            raise RuntimeError("initialization failed")
        return original(self)

    registry.driver.utils = patch._CachedDriverUtils(getter)
    driver = registry.driver()
    with pytest.raises(RuntimeError, match="initialization failed"):
        _ = driver.utils
    assert "utils" not in vars(driver)
    assert driver.utils is driver.utils
    assert len(attempts) == 2


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_resets_initialization_lock(registry):
    patch.patch_flagtree_driver_utils()
    descriptor = registry.driver.utils
    # A worker must not inherit a lock held by a vanished parent thread.
    with descriptor._lock:
        pid = os.fork()
        if pid == 0:
            import signal

            signal.alarm(5)
            driver = registry.driver()
            os._exit(0 if driver.utils is driver.utils else 1)
        _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert registry.calls == []
