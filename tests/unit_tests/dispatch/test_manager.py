# Tests for OpManager: resolution, caching, fallback, fork safety.

import os
import threading
from unittest.mock import MagicMock, patch

import pytest

from sglang_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl
from sglang_fl.dispatch.registry import OpRegistry
from sglang_fl.dispatch.manager import OpManager, get_default_manager, reset_default_manager
from sglang_fl.dispatch.policy import (
    SelectionPolicy,
    get_policy,
    policy_context,
    reset_global_policy,
    set_global_policy,
    with_denied_vendors,
    with_preference,
    with_strict_mode,
    PREFER_DEFAULT,
    PREFER_VENDOR,
    PREFER_REFERENCE,
)


@pytest.fixture
def populated_manager():
    """Manager with pre-registered implementations (bypasses builtin_ops)."""
    registry = OpRegistry()
    manager = OpManager(registry=registry)

    # Manually mark as initialized to skip builtin_ops registration
    manager._state.initialized = True
    manager._state.init_pid = os.getpid()

    # Register test implementations
    def flagos_silu(*a, **kw):
        return "flagos_silu"

    def ref_silu(*a, **kw):
        return "ref_silu"

    def cuda_silu(*a, **kw):
        return "cuda_silu"

    registry.register_many([
        OpImpl(
            op_name="silu_and_mul",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=flagos_silu,
            priority=BackendPriority.DEFAULT,
        ),
        OpImpl(
            op_name="silu_and_mul",
            impl_id="reference.pytorch",
            kind=BackendImplKind.REFERENCE,
            fn=ref_silu,
            priority=BackendPriority.REFERENCE,
        ),
        OpImpl(
            op_name="silu_and_mul",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=cuda_silu,
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        ),
    ])
    return manager


class TestOpManagerResolve:
    def test_resolve_default_prefers_flagos(self, populated_manager):
        reset_global_policy()
        fn = populated_manager.resolve("silu_and_mul")
        assert fn() == "flagos_silu"

    def test_resolve_with_vendor_preference(self, populated_manager):
        with with_preference(PREFER_VENDOR):
            fn = populated_manager.resolve("silu_and_mul")
            assert fn() == "cuda_silu"

    def test_resolve_with_reference_preference(self, populated_manager):
        with with_preference(PREFER_REFERENCE):
            fn = populated_manager.resolve("silu_and_mul")
            assert fn() == "ref_silu"

    def test_resolve_nonexistent_op_raises(self, populated_manager):
        with pytest.raises(RuntimeError, match="No available implementation"):
            populated_manager.resolve("nonexistent_op")

    def test_resolve_with_deny_vendors(self, populated_manager):
        with with_preference(PREFER_VENDOR):
            with with_denied_vendors("cuda"):
                fn = populated_manager.resolve("silu_and_mul")
                # cuda denied, should fall through to flagos (next in order)
                assert fn() != "cuda_silu"


class TestOpManagerCache:
    def test_cache_hit(self, populated_manager):
        reset_global_policy()
        fn1 = populated_manager.resolve("silu_and_mul")
        fn2 = populated_manager.resolve("silu_and_mul")
        assert fn1 is fn2

    def test_cache_invalidated_by_policy_change(self, populated_manager):
        reset_global_policy()
        fn1 = populated_manager.resolve("silu_and_mul")
        assert fn1() == "flagos_silu"

        # Change policy → different cache key
        with with_preference(PREFER_VENDOR):
            fn2 = populated_manager.resolve("silu_and_mul")
            assert fn2() == "cuda_silu"

    def test_cache_invalidated_by_epoch_bump(self, populated_manager):
        reset_global_policy()
        fn1 = populated_manager.resolve("silu_and_mul")
        populated_manager.bump_policy_epoch()
        fn2 = populated_manager.resolve("silu_and_mul")
        # After epoch bump, cache is cleared, but same policy → same result
        assert fn1() == fn2()


class TestOpManagerCall:
    def test_call_direct_mode(self, populated_manager):
        reset_global_policy()
        result = populated_manager.call("silu_and_mul")
        assert result == "flagos_silu"

    def test_call_with_fallback(self, populated_manager):
        """When strict=True and primary fails, falls back to next."""
        registry = OpRegistry()
        manager = OpManager(registry=registry)
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()

        call_count = {"primary": 0, "fallback": 0}

        def failing_fn(*a, **kw):
            call_count["primary"] += 1
            raise RuntimeError("primary failed")

        def fallback_fn(*a, **kw):
            call_count["fallback"] += 1
            return "fallback_result"

        registry.register_many([
            OpImpl(
                op_name="test_op",
                impl_id="default.flagos",
                kind=BackendImplKind.DEFAULT,
                fn=failing_fn,
                priority=BackendPriority.DEFAULT,
            ),
            OpImpl(
                op_name="test_op",
                impl_id="reference.pytorch",
                kind=BackendImplKind.REFERENCE,
                fn=fallback_fn,
                priority=BackendPriority.REFERENCE,
            ),
        ])

        with with_strict_mode():
            result = manager.call("test_op")
            assert result == "fallback_result"
            assert call_count["primary"] == 1
            assert call_count["fallback"] == 1

    def test_call_all_fail_raises(self, populated_manager):
        registry = OpRegistry()
        manager = OpManager(registry=registry)
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()

        registry.register_impl(OpImpl(
            op_name="bad_op",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("fail")),
            priority=BackendPriority.DEFAULT,
        ))

        with with_strict_mode():
            with pytest.raises(RuntimeError, match="All implementations failed"):
                manager.call("bad_op")


class TestOpManagerForkSafety:
    def test_reset_after_fork(self, populated_manager):
        # Simulate fork by calling _reset_after_fork
        populated_manager.resolve("silu_and_mul")  # Populate cache
        assert len(populated_manager._dispatch_cache) > 0

        populated_manager._reset_after_fork()

        assert populated_manager._dispatch_cache == {}
        assert populated_manager._state.initialized is False
        assert populated_manager._called_ops == {}


class TestDefaultManagerSingleton:
    def test_get_default_manager_returns_same_instance(self):
        reset_default_manager()
        m1 = get_default_manager()
        m2 = get_default_manager()
        assert m1 is m2

    def test_reset_default_manager(self):
        reset_default_manager()
        m1 = get_default_manager()
        reset_default_manager()
        m2 = get_default_manager()
        assert m1 is not m2
