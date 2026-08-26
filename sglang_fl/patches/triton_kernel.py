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

"""Reusable Triton kernel launch-meta overrides."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class KernelLaunchMetaProxy:
    """Override selected launch parameters while preserving the kernel API."""

    def __init__(self, kernel: Any, launch_overrides: Mapping[str, Any]) -> None:
        if not launch_overrides:
            raise ValueError("At least one kernel launch override is required.")
        self._kernel = kernel
        self._launch_overrides = dict(launch_overrides)

    @property
    def launch_overrides(self) -> dict[str, Any]:
        return self._launch_overrides.copy()

    def __getitem__(self, grid: Any) -> Any:
        launch = self._kernel[grid]

        def launch_with_overrides(*args: Any, **kwargs: Any) -> Any:
            kwargs.update(self._launch_overrides)
            return launch(*args, **kwargs)

        return launch_with_overrides

    def __getattr__(self, name: str) -> Any:
        return getattr(self._kernel, name)


def patch_kernel_launch_meta(
    module: Any,
    kernel_name: str,
    launch_overrides: Mapping[str, Any],
) -> None:
    """Apply launch overrides to a module's kernel attribute exactly once."""

    kernel = getattr(module, kernel_name)
    if isinstance(kernel, KernelLaunchMetaProxy):
        expected = dict(launch_overrides)
        if kernel.launch_overrides != expected:
            raise RuntimeError(
                f"Kernel {kernel_name} already has launch overrides "
                f"{kernel.launch_overrides}, expected {expected}."
            )
        return
    setattr(module, kernel_name, KernelLaunchMetaProxy(kernel, launch_overrides))
