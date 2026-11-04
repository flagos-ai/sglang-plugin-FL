# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SGLang 0.5.18 import declarations; all kernel calls fail explicitly."""

from _iluvatar_empty_compat import bind_unavailable as _bind_unavailable

__all__ = []

_bind_unavailable(
    globals(),
    (
        "flash_mla_sparse_fwd",
        "flash_mla_with_kvcache",
        "get_mla_metadata",
    ),
)
del _bind_unavailable


class FlashMLASchedMeta:
    """An importable annotation; constructing NVIDIA MLA metadata is unsupported."""

    def __new__(cls, *args, **kwargs):
        raise RuntimeError(
            "sgl_kernel.flash_mla.FlashMLASchedMeta is unavailable in the "
            "Iluvatar empty runtime; select an implemented attention backend."
        )


__all__.append("FlashMLASchedMeta")
