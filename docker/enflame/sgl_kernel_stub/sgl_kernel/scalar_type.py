# Submodule of the sgl_kernel import-time stub — see sgl_kernel/__init__.py.
#
# Unlike the other stub submodules, this one deliberately FAILS to import:
# sglang wraps `from sgl_kernel.scalar_type import ScalarType, scalar_types`
# (sglang/srt/layers/quantization/utils.py::get_scalar_types) in
# try/except ImportError and falls back to its built-in MockScalarTypes. A
# successful stub import would preempt that fallback and then crash on
# `scalar_types.uint4b8` attribute access at module scope.

raise ImportError(
    "sgl_kernel stub: scalar_type intentionally unavailable on this platform; "
    "letting sglang fall back to its built-in MockScalarTypes"
)
