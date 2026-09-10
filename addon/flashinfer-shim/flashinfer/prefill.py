"""Iluvatar/corex flashinfer stub — import face only, no kernels.

See addon/flashinfer-shim/README.md. Nothing here computes anything: corex has
no flashinfer build, so every stubbed call raises instead of silently
producing a wrong result.
"""


from flashinfer import __getattr__  # noqa: F401  (unknown names raise)

def cudnn_batch_prefill_with_kv_cache(*args, **kwargs):
    """Stub: corex has no flashinfer build."""
    raise RuntimeError(
        "prefill.cudnn_batch_prefill_with_kv_cache was called on a platform with no flashinfer "
        "build — this call belongs behind is_flashinfer_available()"
    )
