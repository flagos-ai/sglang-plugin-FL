# Submodule of the sgl_kernel import-time stub — see sgl_kernel/__init__.py.

def __getattr__(name: str):
    if name in ("__path__", "__spec__", "__all__", "__file__"):
        raise AttributeError(name)

    def _raiser(*args, **kwargs):
        raise NotImplementedError(
            f"sgl_kernel.<submodule>.{name} is a CUDA kernel stub; this platform "
            "must route the op through sglang_fl dispatch "
            "(flagos/vendor/reference)."
        )

    return _raiser
