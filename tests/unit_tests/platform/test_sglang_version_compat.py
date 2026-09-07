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


def test_nvidia_uses_sglang_cuda_piecewise_backend() -> None:
    from sglang.srt.compilation.cuda_piecewise_backend import CUDAPiecewiseBackend

    platform = _platform("nvidia", "cuda")

    assert platform.support_piecewise_cuda_graph() is True
    assert platform.get_piecewise_backend_cls() is CUDAPiecewiseBackend


@pytest.mark.parametrize(
    ("vendor", "device_type"),
    [("ascend", "npu"), ("mthreads", "musa")],
)
def test_non_nvidia_piecewise_backend_remains_unsupported(
    vendor: str, device_type: str
) -> None:
    platform = _platform(vendor, device_type)

    assert platform.support_piecewise_cuda_graph() is False
    with pytest.raises(NotImplementedError):
        platform.get_piecewise_backend_cls()


def test_non_cuda_upgrade_targets_keep_legacy_oot_key() -> None:
    assert _platform("mthreads", "musa").get_dispatch_key_name() == "oot"
    assert _platform("ascend", "npu").get_dispatch_key_name() == "oot"


def test_pin_memory_signature_accepts_device() -> None:
    platform = _platform("nvidia", "cuda")
    assert platform.is_pin_memory_available() is True
    assert platform.is_pin_memory_available("cuda") is True
    assert platform.is_pin_memory_available("cpu") is False


@pytest.mark.parametrize(
    ("vendor", "device_type", "torch_backend"),
    [
        ("nvidia", "cuda", "nccl"),
        ("ascend", "npu", "hccl"),
        ("mthreads", "musa", "mccl"),
    ],
)
def test_flagcx_uses_vendor_backend_for_c10d_bootstrap(
    vendor: str, device_type: str, torch_backend: str
) -> None:
    from sglang_fl.distributed.communicator import CommunicatorFL

    platform = _platform(vendor, device_type)
    platform._dist_backend = "flagcx"

    assert platform.get_torch_distributed_backend_str() == torch_backend
    assert platform.get_communicator_class() is CommunicatorFL


def test_native_dist_backend_is_returned_unchanged() -> None:
    platform = _platform("nvidia", "cuda")
    platform._dist_backend = "nccl"

    assert platform.get_torch_distributed_backend_str() == "nccl"


def test_dsa_and_legacy_nsa_factories_are_aliases(monkeypatch) -> None:
    memory_pool = ModuleType("sglang.srt.mem_cache.memory_pool")

    class DSATokenToKVPool:
        pass

    memory_pool.DSATokenToKVPool = DSATokenToKVPool
    monkeypatch.setitem(sys.modules, memory_pool.__name__, memory_pool)

    platform = _platform("nvidia", "cuda")
    assert platform.get_dsa_kv_pool_cls() is platform.get_nsa_kv_pool_cls()
