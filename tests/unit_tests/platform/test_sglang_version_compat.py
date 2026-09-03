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

"""Compatibility checks for the staged SGLang 0.5.18 CUDA upgrade."""

import sys
from types import ModuleType

import pytest

# Import SGLang first, matching its real entry-point lifecycle. Importing the
# plugin platform module directly while ``sglang.__init__`` is still pending
# reverses the platform-discovery order and can create an artificial cycle.
pytest.importorskip("sglang")

from sglang_fl.platform import PlatformFL


def _platform(vendor: str, device_type: str):
    platform = PlatformFL.__new__(PlatformFL)
    platform._vendor_name = vendor
    platform._device_type = device_type
    return platform


def test_nvidia_uses_cuda_fused_op_fallback_key() -> None:
    assert _platform("nvidia", "cuda").get_dispatch_key_name() == "cuda"


def test_non_cuda_upgrade_targets_keep_legacy_oot_key() -> None:
    assert _platform("mthreads", "musa").get_dispatch_key_name() == "oot"
    assert _platform("ascend", "npu").get_dispatch_key_name() == "oot"


def test_pin_memory_signature_accepts_device() -> None:
    platform = _platform("nvidia", "cuda")
    assert platform.is_pin_memory_available() is True
    assert platform.is_pin_memory_available("cuda") is True
    assert platform.is_pin_memory_available("cpu") is False


def test_dsa_and_legacy_nsa_factories_are_aliases(monkeypatch) -> None:
    memory_pool = ModuleType("sglang.srt.mem_cache.memory_pool")

    class DSATokenToKVPool:
        pass

    memory_pool.DSATokenToKVPool = DSATokenToKVPool
    monkeypatch.setitem(sys.modules, memory_pool.__name__, memory_pool)

    platform = _platform("nvidia", "cuda")
    assert platform.get_dsa_kv_pool_cls() is platform.get_nsa_kv_pool_cls()
