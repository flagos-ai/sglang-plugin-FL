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

    # Ops served by TXDA-specific code. Other ops are left to default.flagos /
    # reference.torch: the former stubs (silu_and_mul, rms_norm,
    # rotary_embedding, mrotary_embedding) only raised NotImplementedError,
    # topk is served by default.flagos, fused_recurrent_gated_delta_rule has no
    # SGLang 0.5.18 caller, and fused_recurrent_gated_delta_rule_packed_decode
    # is excluded from FLA dispatch on tsingmicro (SGLang's kernel is called
    # directly, see sglang_fl.load_plugin).
    impls = [
        OpImpl(
            op_name=name,
            impl_id="vendor.tsingmicro",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(getattr(backend, name), is_avail),
            vendor="tsingmicro",
            priority=BackendPriority.VENDOR,
        )
        for name in ("gemma_rms_norm", "fused_moe", "chunk_gated_delta_rule")
    ]

    registry.register_many(impls)
