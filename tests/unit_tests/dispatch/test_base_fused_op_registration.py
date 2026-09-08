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

"""Tests for SGLang 0.5.18 BaseFusedOp OOT registration."""

import pytest

import sglang_fl


def test_base_fused_op_registration_uses_active_platform_key(monkeypatch) -> None:
    try:
        from sglang.kernels.fused_op import BaseFusedOp
        from sglang.srt.platforms import current_platform
    except ImportError:
        pytest.skip("BaseFusedOp is available in the SGLang 0.5.16+ environment")

    class TestOp(BaseFusedOp):
        def forward_native(self, value):
            return value

    class TestDerivedOp(TestOp):
        pass

    class TestSpecificOp(TestOp):
        pass

    class TestSpecificDerivedOp(TestSpecificOp):
        pass

    def bridge(self, value):
        return value + 1

    def specific_bridge(self, value):
        return value + 2

    monkeypatch.setattr(BaseFusedOp, "_oot_forward_registry", {})
    monkeypatch.setattr(
        sglang_fl,
        "_get_bridge_map",
        lambda: {TestOp: bridge, TestSpecificOp: specific_bridge},
    )

    assert sglang_fl._register_base_fused_op_forwards({}) is True

    platform_key = current_platform.get_dispatch_key_name()
    registry = BaseFusedOp._oot_forward_registry[platform_key]
    assert registry[TestOp] is bridge
    assert registry[TestDerivedOp] is bridge
    assert registry[TestSpecificOp] is specific_bridge
    assert registry[TestSpecificDerivedOp] is specific_bridge


def test_base_fused_op_registration_respects_blacklist(monkeypatch) -> None:
    try:
        from sglang.kernels.fused_op import BaseFusedOp
    except ImportError:
        pytest.skip("BaseFusedOp is available in the SGLang 0.5.16+ environment")

    class BlockedOp(BaseFusedOp):
        def forward_native(self, value):
            return value

    monkeypatch.setattr(BaseFusedOp, "_oot_forward_registry", {})
    monkeypatch.setattr(sglang_fl, "_get_bridge_map", lambda: {BlockedOp: lambda: None})

    assert (
        sglang_fl._register_base_fused_op_forwards({"oot_blacklist": ["BlockedOp"]})
        is True
    )
    assert not any(
        BlockedOp in entries for entries in BaseFusedOp._oot_forward_registry.values()
    )


def test_base_fused_op_resolver_handles_subclass_imported_after_setup(
    monkeypatch,
) -> None:
    try:
        from sglang.kernels.fused_op import BaseFusedOp
    except ImportError:
        pytest.skip("BaseFusedOp is available in the SGLang 0.5.16+ environment")

    class RegisteredOp(BaseFusedOp):
        def forward_native(self, value):
            return value

    def bridge(self, value):
        return value + 1

    monkeypatch.setattr(sglang_fl, "_get_bridge_map", lambda: {RegisteredOp: bridge})
    resolve_hook = sglang_fl._make_base_fused_resolve_hook({})

    # This class is deliberately defined after plugin setup, matching model
    # modules that SGLang imports only after load_plugins() has returned.
    class LateImportedOp(RegisteredOp):
        pass

    op = LateImportedOp()
    resolved = resolve_hook(lambda _self: op.forward_native, op)
    assert resolved(41) == 42
