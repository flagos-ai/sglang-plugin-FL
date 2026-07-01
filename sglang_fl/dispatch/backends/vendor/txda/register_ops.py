# TXDA (TsingMicro) backend operator registrations.

from __future__ import annotations

import functools

from sglang_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    """Register all TXDA (VENDOR) operator implementations."""
    from .txda import TxdaBackend

    backend = TxdaBackend()
    is_avail = backend.is_available

    impls = [
        OpImpl(
            op_name="silu_and_mul",
            impl_id="vendor.txda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.silu_and_mul, is_avail),
            vendor="tsingmicro",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="rms_norm",
            impl_id="vendor.txda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rms_norm, is_avail),
            vendor="tsingmicro",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="rotary_embedding",
            impl_id="vendor.txda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rotary_embedding, is_avail),
            vendor="tsingmicro",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="topk",
            impl_id="vendor.txda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.topk, is_avail),
            vendor="tsingmicro",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="gemma_rms_norm",
            impl_id="vendor.txda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.gemma_rms_norm, is_avail),
            vendor="tsingmicro",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="mrotary_embedding",
            impl_id="vendor.txda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.mrotary_embedding, is_avail),
            vendor="tsingmicro",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="fused_moe",
            impl_id="vendor.txda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.fused_moe, is_avail),
            vendor="tsingmicro",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="chunk_gated_delta_rule",
            impl_id="vendor.txda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.chunk_gated_delta_rule, is_avail),
            vendor="tsingmicro",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="fused_recurrent_gated_delta_rule",
            impl_id="vendor.txda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.fused_recurrent_gated_delta_rule, is_avail),
            vendor="tsingmicro",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="fused_recurrent_gated_delta_rule_packed_decode",
            impl_id="vendor.txda",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(
                backend.fused_recurrent_gated_delta_rule_packed_decode, is_avail
            ),
            vendor="tsingmicro",
            priority=BackendPriority.VENDOR,
        ),
    ]

    registry.register_many(impls)
