# Installed into site-packages by docker/hygon/runtime-0.5.18.containerfile.
# Stub only: every attribute resolves to a callable that raises.
def _unavailable(name):
    def _raise(*args, **kwargs):
        raise RuntimeError(
            f"sgl_kernel.{name} is a stub: HCU ships no sgl_kernel wheel"
        )

    return _raise


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return _unavailable(name)
