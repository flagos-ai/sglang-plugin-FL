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

# Template backend operator registrations.
#
# Copy this file as your starting point for a new vendor backend.
# Replace 'template' with your vendor name throughout.

from __future__ import annotations

import functools


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    """Register all Template (VENDOR) operator implementations."""
    from .template import TemplateBackend  # noqa: F401

    # TODO: Add your operator implementations here
    impls = [
        # OpImpl(
        #     op_name="silu_and_mul",
        #     impl_id="vendor.template",
        #     kind=BackendImplKind.VENDOR,
        #     fn=_bind_is_available(backend.silu_and_mul, is_avail),
        #     vendor="template",
        #     priority=BackendPriority.VENDOR,
        # ),
    ]

    registry.register_many(impls)
