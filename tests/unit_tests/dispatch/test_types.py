# Tests for dispatch type definitions.

import pytest

from sglang_fl.dispatch.types import (
    BackendImplKind,
    BackendPriority,
    OpImpl,
    match_token,
)


class TestBackendImplKind:
    def test_enum_values(self):
        assert BackendImplKind.DEFAULT.value == "flagos"
        assert BackendImplKind.REFERENCE.value == "reference"
        assert BackendImplKind.VENDOR.value == "vendor"

    def test_str_representation(self):
        assert str(BackendImplKind.DEFAULT) == "flagos"
        assert str(BackendImplKind.REFERENCE) == "reference"
        assert str(BackendImplKind.VENDOR) == "vendor"


class TestBackendPriority:
    def test_priority_ordering(self):
        assert BackendPriority.DEFAULT > BackendPriority.VENDOR
        assert BackendPriority.VENDOR > BackendPriority.REFERENCE
        assert BackendPriority.DEFAULT > BackendPriority.REFERENCE

    def test_spacing(self):
        # Priorities spaced by 50 for future insertion
        assert BackendPriority.DEFAULT - BackendPriority.VENDOR == 50
        assert BackendPriority.VENDOR - BackendPriority.REFERENCE == 50


class TestOpImpl:
    def test_create_default_impl(self):
        fn = lambda x: x
        impl = OpImpl(
            op_name="silu_and_mul",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=fn,
            priority=BackendPriority.DEFAULT,
        )
        assert impl.op_name == "silu_and_mul"
        assert impl.impl_id == "default.flagos"
        assert impl.kind == BackendImplKind.DEFAULT
        assert impl.vendor is None
        assert impl.priority == BackendPriority.DEFAULT

    def test_create_vendor_impl_requires_vendor_name(self):
        fn = lambda x: x
        with pytest.raises(ValueError, match="must specify vendor name"):
            OpImpl(
                op_name="silu_and_mul",
                impl_id="vendor.cuda",
                kind=BackendImplKind.VENDOR,
                fn=fn,
            )

    def test_create_vendor_impl_with_vendor_name(self):
        fn = lambda x: x
        impl = OpImpl(
            op_name="silu_and_mul",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=fn,
            vendor="cuda",
            priority=BackendPriority.VENDOR,
        )
        assert impl.vendor == "cuda"
        assert impl.kind == BackendImplKind.VENDOR

    def test_is_available_default_true(self):
        fn = lambda x: x
        impl = OpImpl(
            op_name="silu_and_mul",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=fn,
        )
        assert impl.is_available() is True

    def test_is_available_with_checker_true(self):
        def fn(x):
            return x
        fn._is_available = lambda: True
        impl = OpImpl(
            op_name="silu_and_mul",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=fn,
        )
        assert impl.is_available() is True

    def test_is_available_with_checker_false(self):
        def fn(x):
            return x
        fn._is_available = lambda: False
        impl = OpImpl(
            op_name="silu_and_mul",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=fn,
        )
        assert impl.is_available() is False

    def test_is_available_handles_exception(self):
        def fn(x):
            return x
        fn._is_available = lambda: 1 / 0
        impl = OpImpl(
            op_name="silu_and_mul",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=fn,
        )
        assert impl.is_available() is False

    def test_frozen_dataclass(self):
        fn = lambda x: x
        impl = OpImpl(
            op_name="silu_and_mul",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=fn,
        )
        with pytest.raises(AttributeError):
            impl.op_name = "new_name"

    def test_supported_dtypes(self):
        fn = lambda x: x
        impl = OpImpl(
            op_name="silu_and_mul",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=fn,
            supported_dtypes={"float16", "bfloat16"},
        )
        assert "float16" in impl.supported_dtypes
        assert "float32" not in impl.supported_dtypes


class TestMatchToken:
    @pytest.fixture
    def default_impl(self):
        return OpImpl(
            op_name="silu_and_mul",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=lambda x: x,
        )

    @pytest.fixture
    def reference_impl(self):
        return OpImpl(
            op_name="silu_and_mul",
            impl_id="reference.pytorch",
            kind=BackendImplKind.REFERENCE,
            fn=lambda x: x,
        )

    @pytest.fixture
    def vendor_impl(self):
        return OpImpl(
            op_name="silu_and_mul",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=lambda x: x,
            vendor="cuda",
        )

    def test_match_flagos_token(self, default_impl, reference_impl, vendor_impl):
        assert match_token(default_impl, "flagos") is True
        assert match_token(reference_impl, "flagos") is False
        assert match_token(vendor_impl, "flagos") is False

    def test_match_reference_token(self, default_impl, reference_impl, vendor_impl):
        assert match_token(default_impl, "reference") is False
        assert match_token(reference_impl, "reference") is True
        assert match_token(vendor_impl, "reference") is False

    def test_match_vendor_token(self, default_impl, reference_impl, vendor_impl):
        assert match_token(default_impl, "vendor") is False
        assert match_token(reference_impl, "vendor") is False
        assert match_token(vendor_impl, "vendor") is True

    def test_match_vendor_specific(self, vendor_impl):
        assert match_token(vendor_impl, "vendor:cuda") is True
        assert match_token(vendor_impl, "vendor:ascend") is False

    def test_match_impl_id(self, default_impl, reference_impl):
        assert match_token(default_impl, "impl:default.flagos") is True
        assert match_token(default_impl, "impl:reference.pytorch") is False
        assert match_token(reference_impl, "impl:reference.pytorch") is True

    def test_unknown_token_returns_false(self, default_impl):
        assert match_token(default_impl, "unknown") is False
        assert match_token(default_impl, "") is False
