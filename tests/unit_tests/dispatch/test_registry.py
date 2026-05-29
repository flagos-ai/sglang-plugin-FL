# Tests for OpRegistry thread safety and correctness.

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from sglang_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl
from sglang_fl.dispatch.registry import OpRegistry


class TestOpRegistryBasic:
    def test_register_and_get(self, registry, make_impl):
        impl = make_impl(op_name="silu_and_mul", impl_id="default.flagos")
        registry.register_impl(impl)
        impls = registry.get_implementations("silu_and_mul")
        assert len(impls) == 1
        assert impls[0].impl_id == "default.flagos"

    def test_register_many(self, registry, make_impl):
        impls = [
            make_impl(op_name="silu_and_mul", impl_id="default.flagos"),
            make_impl(op_name="silu_and_mul", impl_id="reference.pytorch", kind=BackendImplKind.REFERENCE),
            make_impl(op_name="rms_norm", impl_id="default.flagos"),
        ]
        registry.register_many(impls)
        assert len(registry.get_implementations("silu_and_mul")) == 2
        assert len(registry.get_implementations("rms_norm")) == 1

    def test_duplicate_impl_id_raises(self, registry, make_impl):
        impl1 = make_impl(op_name="silu_and_mul", impl_id="default.flagos")
        impl2 = make_impl(op_name="silu_and_mul", impl_id="default.flagos")
        registry.register_impl(impl1)
        with pytest.raises(ValueError, match="Duplicate impl_id"):
            registry.register_impl(impl2)

    def test_get_nonexistent_op(self, registry):
        assert registry.get_implementations("nonexistent") == []

    def test_get_implementation_by_id(self, registry, make_impl):
        impl = make_impl(op_name="silu_and_mul", impl_id="default.flagos")
        registry.register_impl(impl)
        found = registry.get_implementation("silu_and_mul", "default.flagos")
        assert found is impl
        assert registry.get_implementation("silu_and_mul", "nonexistent") is None

    def test_list_operators(self, registry, make_impl):
        registry.register_impl(make_impl(op_name="silu_and_mul", impl_id="a"))
        registry.register_impl(make_impl(op_name="rms_norm", impl_id="b"))
        registry.register_impl(make_impl(op_name="rotary_embedding", impl_id="c"))
        ops = registry.list_operators()
        assert set(ops) == {"silu_and_mul", "rms_norm", "rotary_embedding"}

    def test_clear(self, registry, make_impl):
        registry.register_impl(make_impl(op_name="silu_and_mul", impl_id="a"))
        registry.clear()
        assert registry.list_operators() == []


class TestOpRegistrySnapshot:
    def test_snapshot_is_independent(self, registry, make_impl):
        registry.register_impl(make_impl(op_name="silu_and_mul", impl_id="a"))
        snap = registry.snapshot()
        # Modify registry after snapshot
        registry.register_impl(make_impl(op_name="silu_and_mul", impl_id="b"))
        # Snapshot should not reflect the change
        assert len(snap.impls_by_op["silu_and_mul"]) == 1

    def test_snapshot_contains_all_ops(self, registry, make_impl):
        registry.register_impl(make_impl(op_name="silu_and_mul", impl_id="a"))
        registry.register_impl(make_impl(op_name="rms_norm", impl_id="b"))
        snap = registry.snapshot()
        assert "silu_and_mul" in snap.impls_by_op
        assert "rms_norm" in snap.impls_by_op


class TestOpRegistryThreadSafety:
    def test_concurrent_register(self, make_impl):
        registry = OpRegistry()
        n_threads = 20
        n_ops_per_thread = 50

        def register_batch(thread_id):
            for i in range(n_ops_per_thread):
                impl = make_impl(
                    op_name=f"op_{i}",
                    impl_id=f"impl_{thread_id}_{i}",
                )
                registry.register_impl(impl)

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(register_batch, t) for t in range(n_threads)]
            for f in as_completed(futures):
                f.result()  # Raise if any thread failed

        # Each op should have n_threads implementations
        for i in range(n_ops_per_thread):
            impls = registry.get_implementations(f"op_{i}")
            assert len(impls) == n_threads

    def test_concurrent_read_write(self, make_impl):
        registry = OpRegistry()
        # Pre-populate
        for i in range(10):
            registry.register_impl(make_impl(op_name="shared_op", impl_id=f"init_{i}"))

        errors = []

        def writer(thread_id):
            for i in range(50):
                try:
                    registry.register_impl(
                        make_impl(op_name="shared_op", impl_id=f"w{thread_id}_{i}")
                    )
                except ValueError:
                    pass  # Duplicate is fine

        def reader():
            for _ in range(100):
                impls = registry.get_implementations("shared_op")
                if not impls:
                    errors.append("Got empty list during concurrent access")

        threads = []
        for t in range(5):
            threads.append(threading.Thread(target=writer, args=(t,)))
        for _ in range(5):
            threads.append(threading.Thread(target=reader))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
