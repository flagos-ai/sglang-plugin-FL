# Tests for call_op / resolve_op high-level API and builtin_ops registration.

import os
from unittest.mock import patch

import pytest

from sglang_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl
from sglang_fl.dispatch.registry import OpRegistry
from sglang_fl.dispatch.manager import get_default_manager, reset_default_manager
from sglang_fl.dispatch import call_op, resolve_op
from sglang_fl.dispatch.policy import (
    reset_global_policy,
    with_preference,
    PREFER_VENDOR,
)


class TestCallOp:
    """Test the top-level call_op / resolve_op convenience functions."""

    @pytest.fixture(autouse=True)
    def _reset_manager(self):
        reset_default_manager()
        yield
        reset_default_manager()

    def test_call_op_uses_default_manager(self):
        """call_op delegates to get_default_manager().call()."""
        manager = get_default_manager()
        registry = manager.registry

        # Force initialization
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()
        registry.clear()

        def test_fn(*a, **kw):
            return "test_result"

        registry.register_impl(
            OpImpl(
                op_name="test_op",
                impl_id="default.flagos",
                kind=BackendImplKind.DEFAULT,
                fn=test_fn,
                priority=BackendPriority.DEFAULT,
            )
        )

        result = call_op("test_op")
        assert result == "test_result"

    def test_resolve_op_returns_callable(self):
        manager = get_default_manager()
        registry = manager.registry

        manager._state.initialized = True
        manager._state.init_pid = os.getpid()
        registry.clear()

        def test_fn(*a, **kw):
            return "resolved"

        registry.register_impl(
            OpImpl(
                op_name="test_op",
                impl_id="default.flagos",
                kind=BackendImplKind.DEFAULT,
                fn=test_fn,
                priority=BackendPriority.DEFAULT,
            )
        )

        fn = resolve_op("test_op")
        assert callable(fn)
        assert fn() == "resolved"


class TestBuiltinOpsRegistration:
    """Test that builtin_ops.register_builtins correctly discovers backends."""

    def test_register_builtins_populates_registry(self):
        from sglang_fl.dispatch.builtin_ops import register_builtins

        registry = OpRegistry()
        register_builtins(registry)

        ops = registry.list_operators()
        # Should have at least some operators registered
        assert len(ops) > 0
        # silu_and_mul and rms_norm should always be present (reference at minimum)
        assert "silu_and_mul" in ops
        assert "rms_norm" in ops

    def test_register_builtins_has_reference_impls(self):
        from sglang_fl.dispatch.builtin_ops import register_builtins

        registry = OpRegistry()
        register_builtins(registry)

        # Reference implementations should always be available
        silu_impls = registry.get_implementations("silu_and_mul")
        ref_impls = [i for i in silu_impls if i.kind == BackendImplKind.REFERENCE]
        assert len(ref_impls) >= 1

    def test_reference_backend_only_registers_torch_fallbacks(self):
        """Upstream Triton adapters must not masquerade as generic references."""
        from sglang_fl.dispatch.backends.reference.register_ops import (
            register_builtins as register_reference,
        )

        registry = OpRegistry()
        register_reference(registry)

        implementations = [
            impl
            for op_name in registry.list_operators()
            for impl in registry.get_implementations(op_name)
        ]
        assert implementations
        assert {impl.impl_id for impl in implementations} == {"reference.torch"}

        unsupported_reference_ops = {
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule",
            "fused_recurrent_gated_delta_rule_packed_decode",
            "fused_moe",
        }
        assert unsupported_reference_ops.isdisjoint(registry.list_operators())

    def test_cuda_backend_owns_upstream_triton_adapters(self):
        """FLA and MoE use the explicit CUDA vendor path until FlagOS covers them."""
        from sglang_fl.dispatch.backends.vendor.cuda.register_ops import (
            register_builtins as register_cuda,
        )

        registry = OpRegistry()
        register_cuda(registry)

        transitional_ops = {
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule",
            "fused_recurrent_gated_delta_rule_packed_decode",
            "fused_moe",
        }
        for op_name in transitional_ops:
            implementations = registry.get_implementations(op_name)
            assert len(implementations) == 1
            assert implementations[0].impl_id == "vendor.cuda"
            assert implementations[0].kind == BackendImplKind.VENDOR
            assert implementations[0].vendor == "cuda"

    def test_vendor_discovery_skips_template(self):
        """The 'template' vendor directory should be skipped."""
        from sglang_fl.dispatch.builtin_ops import register_builtins

        registry = OpRegistry()
        register_builtins(registry)

        # No impl should have vendor="template"
        for op in registry.list_operators():
            for impl in registry.get_implementations(op):
                if impl.vendor:
                    assert impl.vendor != "template"

    def test_vendor_discovery_handles_import_error(self):
        """If a vendor module fails to import, registration continues."""
        from sglang_fl.dispatch.builtin_ops import _register_vendor_backends

        registry = OpRegistry()
        with patch("importlib.import_module", side_effect=ImportError("no module")):
            # Should not raise
            _register_vendor_backends(registry)
        # Registry may be empty but no crash
        assert isinstance(registry.list_operators(), list)


class TestFullPipeline:
    """Integration test: register → resolve → call through the full dispatch stack."""

    @pytest.fixture(autouse=True)
    def _reset(self):
        reset_default_manager()
        reset_global_policy()
        yield
        reset_default_manager()
        reset_global_policy()

    def test_full_pipeline_default_preference(self):
        manager = get_default_manager()
        registry = manager.registry
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()
        registry.clear()

        results = []

        def flagos_fn(*a, **kw):
            results.append("flagos")
            return "flagos"

        def vendor_fn(*a, **kw):
            results.append("vendor")
            return "vendor"

        registry.register_many(
            [
                OpImpl(
                    op_name="pipeline_op",
                    impl_id="default.flagos",
                    kind=BackendImplKind.DEFAULT,
                    fn=flagos_fn,
                    priority=BackendPriority.DEFAULT,
                ),
                OpImpl(
                    op_name="pipeline_op",
                    impl_id="vendor.cuda",
                    kind=BackendImplKind.VENDOR,
                    fn=vendor_fn,
                    vendor="cuda",
                    priority=BackendPriority.VENDOR,
                ),
            ]
        )

        # Default preference → flagos
        result = call_op("pipeline_op")
        assert result == "flagos"

        # Switch to vendor preference
        with with_preference(PREFER_VENDOR):
            result = call_op("pipeline_op")
            assert result == "vendor"

        assert results == ["flagos", "vendor"]
