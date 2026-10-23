# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded import declarations, never a substitute for accelerator kernels."""


def unavailable(module, name):
    """Create a named function that refuses every attempted kernel call."""

    def unsupported(*args, **kwargs):
        raise RuntimeError(
            f"{module}.{name} is unavailable in the Iluvatar empty runtime. "
            "Use an implemented FlagGems, Triton, or PyTorch backend. "
            "sglang-iluvatar-compat supplies import compatibility only."
        )

    unsupported.__name__ = name
    unsupported.__qualname__ = name
    unsupported.__module__ = module
    unsupported.__doc__ = "Unavailable accelerator kernel; calling it always raises."
    return unsupported


def bind_unavailable(namespace, names):
    """Install exactly the reviewed names; unknown attributes stay absent."""
    namespace["__all__"] = list(names)
    namespace["__sglang_iluvatar_empty_compat__"] = True
    namespace["__flagos_import_shim__"] = True
    for name in names:
        namespace[name] = unavailable(namespace["__name__"], name)
