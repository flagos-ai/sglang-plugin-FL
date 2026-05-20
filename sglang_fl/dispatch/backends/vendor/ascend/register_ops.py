# Ascend backend operator registrations.

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
    """Register all Ascend (VENDOR) operator implementations."""
    from .ascend import AscendBackend

    backend = AscendBackend()
    is_avail = backend.is_available

    impls = [
        OpImpl(
            op_name="silu_and_mul",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.silu_and_mul, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="rms_norm",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rms_norm, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="rotary_embedding",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rotary_embedding, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="rotary_embedding_with_kv_cache",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rotary_embedding_with_kv_cache, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
    ]

    registry.register_many(impls)
