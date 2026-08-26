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

from types import SimpleNamespace

import pytest

from sglang_fl.patches.triton_kernel import (
    KernelLaunchMetaProxy,
    patch_kernel_launch_meta,
)


class _FakeKernel:
    marker = "wrapped"

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            return grid, args, kwargs

        return launch


def test_kernel_launch_meta_proxy_overrides_multiple_parameters():
    proxy = KernelLaunchMetaProxy(
        _FakeKernel(),
        {"BLOCK_N": 2048, "num_warps": 4, "num_stages": 2},
    )

    grid, args, kwargs = proxy["grid"](
        1,
        BLOCK_N=256,
        num_warps=8,
        other=True,
    )

    assert grid == "grid"
    assert args == (1,)
    assert kwargs == {
        "BLOCK_N": 2048,
        "num_warps": 4,
        "num_stages": 2,
        "other": True,
    }
    assert proxy.marker == "wrapped"


def test_patch_kernel_launch_meta_is_idempotent():
    module = SimpleNamespace(kernel=_FakeKernel())
    overrides = {"num_warps": 8}

    patch_kernel_launch_meta(module, "kernel", overrides)
    proxy = module.kernel
    patch_kernel_launch_meta(module, "kernel", overrides)

    assert module.kernel is proxy


def test_patch_kernel_launch_meta_rejects_conflicting_overrides():
    module = SimpleNamespace(kernel=_FakeKernel())
    patch_kernel_launch_meta(module, "kernel", {"num_warps": 8})

    with pytest.raises(RuntimeError, match="already has launch overrides"):
        patch_kernel_launch_meta(module, "kernel", {"num_warps": 4})
