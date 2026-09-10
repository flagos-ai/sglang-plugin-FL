"""Iluvatar/corex flashinfer stub — import face only, no kernels.

See addon/flashinfer-shim/README.md. Nothing here computes anything: corex has
no flashinfer build, so every stubbed call raises instead of silently
producing a wrong result.
"""



def _unavailable(name):
    def _raise(*args, **kwargs):
        raise RuntimeError(
            f"{__name__}.{name} was called on a platform with no flashinfer "
            f"build — this call belongs behind is_flashinfer_available()"
        )

    _raise.__name__ = name
    return _raise


_ALLOWED = ('bmm_fp8',)


def __getattr__(name):
    if name in _ALLOWED:
        return _unavailable(name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r} "
        f"(flashinfer stub: only {sorted(_ALLOWED)} is stubbed)"
    )


for _n in _ALLOWED:
    globals()[_n] = _unavailable(_n)
del _n


def __dir__():
    return sorted(list(globals()) + list(_ALLOWED))
