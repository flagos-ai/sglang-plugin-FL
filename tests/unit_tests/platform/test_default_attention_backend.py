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

"""Default attention backend per vendor (PlatformFL.get_default_attention_backend)."""

from types import SimpleNamespace

import pytest

pytest.importorskip("sglang")

from sglang_fl.platform import PlatformFL


def _default_backend(vendor_name: str) -> str:
    platform = PlatformFL.__new__(PlatformFL)
    platform._vendor_name = vendor_name
    return platform.get_default_attention_backend()


def test_amd_defaults_to_the_triton_attention_backend():
    assert _default_backend("amd") == "triton"


def test_unmapped_vendor_falls_back_to_torch_native():
    assert _default_backend("no-such-vendor") == "torch_native"


def test_amd_default_is_filled_into_server_args():
    server_args = SimpleNamespace(attention_backend=None, mm_attention_backend=None)
    platform = PlatformFL.__new__(PlatformFL)
    platform._vendor_name = "amd"
    platform._device_type = "cuda"

    platform.apply_server_args_defaults(server_args)

    assert server_args.attention_backend == "triton"


def test_explicit_attention_backend_is_kept():
    server_args = SimpleNamespace(attention_backend="flashinfer")
    platform = PlatformFL.__new__(PlatformFL)
    platform._vendor_name = "amd"
    platform._device_type = "cuda"

    platform.apply_server_args_defaults(server_args)

    assert server_args.attention_backend == "flashinfer"
