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

# AMD backend operator registrations.

from __future__ import annotations

import functools

from sglang_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl


def _bind_is_available(fn, is_available_fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    from .amd import AmdBackend

    backend = AmdBackend()
    is_avail = backend.is_available

    registry.register_many(
        [
            OpImpl(
                op_name=op_name,
                impl_id="vendor.amd",
                kind=BackendImplKind.VENDOR,
                fn=_bind_is_available(getattr(backend, op_name), is_avail),
                vendor="amd",
                priority=BackendPriority.VENDOR,
            )
            for op_name in (
                "fused_moe",
                "chunk_gated_delta_rule",
                "fused_recurrent_gated_delta_rule",
                "fused_recurrent_gated_delta_rule_packed_decode",
            )
        ]
    )
