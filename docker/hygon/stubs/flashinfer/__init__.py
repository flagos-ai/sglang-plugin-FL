# Installed into site-packages by docker/hygon/runtime-0.5.18.containerfile.
# Stub only: every attribute resolves to a callable that raises.
def _unavailable(name):
    def _raise(*args, **kwargs):
        raise RuntimeError(
            f"flashinfer.{name} is a stub: HCU ships no flashinfer wheel"
        )

    return _raise


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return _unavailable(name)
