"""Iluvatar/corex flashinfer stub — import face only, no kernels.

See addon/flashinfer-shim/README.md. Nothing here computes anything: corex has
no flashinfer build, so every stubbed call raises instead of silently
producing a wrong result.
"""


from flashinfer import __getattr__  # noqa: F401  (unknown names raise)

def min_p_sampling_from_probs(*args, **kwargs):
    """Stub: corex has no flashinfer build."""
    raise RuntimeError(
        "sampling.min_p_sampling_from_probs was called on a platform with no flashinfer "
        "build — this call belongs behind is_flashinfer_available()"
    )

def top_k_top_p_sampling_from_probs(*args, **kwargs):
    """Stub: corex has no flashinfer build."""
    raise RuntimeError(
        "sampling.top_k_top_p_sampling_from_probs was called on a platform with no flashinfer "
        "build — this call belongs behind is_flashinfer_available()"
    )
