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

# FlagOS backend operator registrations.

from __future__ import annotations

import functools

from sglang_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    """Register all FlagOS (DEFAULT) operator implementations."""
    from .flagos import FlagOSBackend

    backend = FlagOSBackend()
    is_avail = backend.is_available

    impls = [
        OpImpl(
            op_name="silu_and_mul",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(backend.silu_and_mul, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        OpImpl(
            op_name="rms_norm",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(backend.rms_norm, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        OpImpl(
            op_name="rotary_embedding",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(backend.rotary_embedding, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        OpImpl(
            op_name="topk",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(backend.topk, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
        OpImpl(
            op_name="fused_recurrent_gated_delta_rule",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(backend.fused_recurrent_gated_delta_rule, is_avail),
            vendor=None,
            priority=BackendPriority.DEFAULT,
        ),
    ]

    registry.register_many(impls)
