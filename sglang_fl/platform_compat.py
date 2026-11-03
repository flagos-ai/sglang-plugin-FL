"""Compatibility helpers for SGLang platform interfaces."""

from __future__ import annotations

import inspect


def make_mla_pool_cls(pool_cls: type) -> type:
    """Accept the SGLang 0.5.18 OOT-only ``index_head_dim`` argument.

    SGLang's OOT KV configurator passes ``index_head_dim`` to every MLA pool,
    while the base ``MLATokenToKVPool`` in the same release does not yet accept
    it.  Keep the native class when its signature has caught up; otherwise use
    a narrow adapter for non-DSA MLA models such as Xing4.
    """

    if "index_head_dim" in inspect.signature(pool_cls.__init__).parameters:
        return pool_cls

    class CompatibleMLATokenToKVPool(pool_cls):
        def __init__(self, *args, index_head_dim=None, **kwargs):
            if index_head_dim is not None:
                raise NotImplementedError(
                    "This SGLang MLATokenToKVPool cannot allocate a DSA "
                    "index cache; upgrade SGLang or provide a DSA pool."
                )
            super().__init__(*args, **kwargs)

    CompatibleMLATokenToKVPool.__name__ = f"Compatible{pool_cls.__name__}"
    CompatibleMLATokenToKVPool.__qualname__ = CompatibleMLATokenToKVPool.__name__
    return CompatibleMLATokenToKVPool
