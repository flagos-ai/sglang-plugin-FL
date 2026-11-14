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

"""Compatibility coverage required by HyV4 on recent SGLang releases."""

import inspect

import pytest


def _platform_for(device_type: str):
    from sglang_fl.platform import PlatformFL

    platform = PlatformFL.__new__(PlatformFL)
    platform._device_type = device_type
    return platform


@pytest.mark.parametrize(
    ("device_type", "expected"),
    [
        ("cuda", True),
        ("npu", True),
        ("xpu", True),
        ("musa", True),
        ("tsingmicro", True),
        ("cpu", False),
    ],
)
def test_pin_memory_signature_and_existing_results(device_type, expected):
    from sglang_fl.platform import PlatformFL

    signature = inspect.signature(PlatformFL.is_pin_memory_available)
    assert signature.parameters["device"].default is None
    platform = _platform_for(device_type)
    assert platform.is_pin_memory_available() is expected
    assert platform.is_pin_memory_available(device=device_type) is expected


def test_dsa_kv_pool_uses_sglang_pool():
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

    assert _platform_for("musa").get_dsa_kv_pool_cls() is DSATokenToKVPool


def test_graph_runner_uses_current_sglang_api():
    try:
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )
    except ImportError:
        pytest.skip("current DecodeCudaGraphRunner API is unavailable")

    assert _platform_for("musa").get_graph_runner_cls() is DecodeCudaGraphRunner
