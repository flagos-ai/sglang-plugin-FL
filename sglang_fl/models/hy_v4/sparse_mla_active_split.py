# SPDX-License-Identifier: Apache-2.0
"""Policy-selected routing for the FlagGems-sglang MLA operator.

Indices remain native TopK physical KV slots. Policy, metadata qualification,
and native fallback belong here; numerical kernels belong to FlagGems-sglang.
"""

from .attention_policy import eligible_forward


def _public_provider():
    import sys

    import torch

    module_name = "flaggems_sglang.ops.sparse_mla_active_split"
    if (
        torch.cuda.is_available()
        and torch.cuda.is_current_stream_capturing()
        and module_name not in sys.modules
    ):
        return None
    try:
        import flaggems_sglang
        from flaggems_sglang.ops.sparse_mla_active_split import (
            metadata,
            sparse_mla_active_split,
        )
    except ModuleNotFoundError as error:
        if error.name not in ("flaggems_sglang", "flaggems_sglang.ops", module_name):
            raise
        return None
    public = getattr(flaggems_sglang, "sparse_mla_active_split", None)
    lookup = getattr(flaggems_sglang, "get_op", None)
    if (
        public is not sparse_mla_active_split
        or not callable(lookup)
        or lookup("sparse_mla_active_split") is not public
    ):
        return None
    return (public, metadata)


def maybe_run(
    backend, query, kv_cache, indices, sm_scale, sink, lengths, heads, v_head_dim
):
    if not eligible_forward(backend):
        return None
    if (
        getattr(backend, "device_sm_major", None) != 9
        or getattr(backend, "real_page_size", None) != 64
        or heads != 4
        or (v_head_dim != 512)
        or (query.shape[0] not in (1, 4))
    ):
        return None
    from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

    if not get_is_capture_mode():
        return None
    import torch

    if torch.compiler.is_compiling():
        return None
    from sglang_fl.dispatch import resolve_op

    selected = resolve_op("sparse_mla_active_split")
    return selected(query, kv_cache, indices, sm_scale, sink, lengths, heads)


def _native_route(*args):
    # The caller retains the native FlashMLA call, including its metadata and
    # output ownership. A decline occurs before any numerical execution.
    return None


def _library_route(query, kv_cache, indices, sm_scale, sink, lengths, heads):
    resolved = _public_provider()
    if resolved is None:
        return None
    public, metadata = resolved
    if metadata(query, kv_cache, indices, sm_scale, heads, sink, lengths) is None:
        return None
    return public(
        query,
        kv_cache,
        indices,
        sm_scale,
        active_heads=heads,
        sink=sink,
        lengths=lengths,
    )[0]


def register_ops(registry):
    """Use the shared manager's preference, per-op order and vendor filters.

    Native is the default. Selecting vendor:nvidia or the explicit library
    implementation opts into the optional provider before any launch.
    """
    from sglang_fl.dispatch import BackendImplKind, BackendPriority, OpImpl

    registry.register_many(
        [
            OpImpl(
                "sparse_mla_active_split",
                "hy4.native",
                BackendImplKind.DEFAULT,
                _native_route,
                priority=BackendPriority.DEFAULT,
            ),
            OpImpl(
                "sparse_mla_active_split",
                "hy4.reference",
                BackendImplKind.REFERENCE,
                _native_route,
                priority=BackendPriority.REFERENCE,
            ),
            OpImpl(
                "sparse_mla_active_split",
                "hy4.flaggems",
                BackendImplKind.VENDOR,
                _library_route,
                vendor="nvidia",
                priority=BackendPriority.VENDOR,
            ),
        ]
    )
