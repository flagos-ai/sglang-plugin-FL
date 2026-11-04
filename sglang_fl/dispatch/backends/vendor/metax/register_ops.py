# MetaX/MACA backend operator registrations.

from __future__ import annotations

import functools

from sglang_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl


def _bind_is_available(fn, is_available_fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


@functools.lru_cache(maxsize=1)
def _has_sgl_kernel() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("sgl_kernel") is not None
    except (ImportError, ValueError):
        return False


def register_builtins(registry) -> None:
    from .metax import MetaxBackend

    backend = MetaxBackend()
    is_avail = backend.is_available

    # These MetaX impls delegate to vendor.cuda, which calls sgl_kernel ops. In
    # Empty mode (no sgl_kernel) they report unavailable so dispatch selects the
    # FlagGems / reference implementation directly instead of failing per call.
    def is_avail_sgl_kernel():
        return is_avail() and _has_sgl_kernel()

    impls = [
        OpImpl(
            op_name="silu_and_mul",
            impl_id="vendor.metax",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.silu_and_mul, is_avail_sgl_kernel),
            vendor="metax",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="rms_norm",
            impl_id="vendor.metax",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rms_norm, is_avail_sgl_kernel),
            vendor="metax",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="gemma_rms_norm",
            impl_id="vendor.metax",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.gemma_rms_norm, is_avail_sgl_kernel),
            vendor="metax",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="rotary_embedding",
            impl_id="vendor.metax",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rotary_embedding, is_avail_sgl_kernel),
            vendor="metax",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="topk",
            impl_id="vendor.metax",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.topk, is_avail_sgl_kernel),
            vendor="metax",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="fused_moe",
            impl_id="vendor.metax",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.fused_moe, is_avail),
            vendor="metax",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="chunk_gated_delta_rule",
            impl_id="vendor.metax",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.chunk_gated_delta_rule, is_avail),
            vendor="metax",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="fused_recurrent_gated_delta_rule",
            impl_id="vendor.metax",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.fused_recurrent_gated_delta_rule, is_avail),
            vendor="metax",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="fused_recurrent_gated_delta_rule_packed_decode",
            impl_id="vendor.metax",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(
                backend.fused_recurrent_gated_delta_rule_packed_decode, is_avail
            ),
            vendor="metax",
            priority=BackendPriority.VENDOR,
        ),
    ]

    registry.register_many(impls)
