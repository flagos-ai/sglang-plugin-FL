# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from sglang_fl.dispatch.manager import OpManager
from sglang_fl.dispatch.policy import reset_global_policy, with_denied_vendors
from sglang_fl.dispatch.registry import OpRegistry
from sglang_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl


@pytest.fixture
def manager():
    registry = OpRegistry()
    registry.register_many(
        [
            OpImpl(
                op_name="example",
                impl_id="default.flagos",
                kind=BackendImplKind.DEFAULT,
                fn=lambda: "flagos",
                priority=BackendPriority.DEFAULT,
            ),
            OpImpl(
                op_name="example",
                impl_id="reference.pytorch",
                kind=BackendImplKind.REFERENCE,
                fn=lambda: "reference",
                priority=BackendPriority.REFERENCE,
            ),
            OpImpl(
                op_name="example",
                impl_id="vendor.cuda",
                kind=BackendImplKind.VENDOR,
                fn=lambda: "vendor",
                vendor="cuda",
                priority=BackendPriority.VENDOR,
            ),
        ]
    )
    result = OpManager(registry=registry)
    result._state.initialized = True
    result._state.init_pid = os.getpid()
    return result


@pytest.fixture(autouse=True)
def platform_profile(monkeypatch):
    monkeypatch.setenv("SGLANG_FL_MODE", "platform_profile")
    reset_global_policy()
    yield
    reset_global_policy()


def test_platform_profile_excludes_flagos_and_reference(manager):
    assert manager.resolve("example")() == "vendor"


def test_platform_profile_reports_available_vendor(manager):
    assert manager.has_available_vendor("example")


def test_platform_profile_does_not_select_non_vendor_fallback(manager):
    with with_denied_vendors("cuda"):
        assert not manager.has_available_vendor("example")
        with pytest.raises(RuntimeError, match="No available implementation"):
            manager.resolve("example")
