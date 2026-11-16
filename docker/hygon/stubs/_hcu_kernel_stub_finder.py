# Installed into site-packages by docker/hygon/runtime-0.5.18.containerfile.
# Serves any sgl_kernel.* / flashinfer.* submodule import with a
# module whose attributes raise on call; the file stubs above keep
# winning for the packages that exist on disk.
import importlib.abc
import importlib.util
import sys
import types

_STUB_ROOTS = ("sgl_kernel", "flashinfer")


def _unavailable(name):
    def _raise(*args, **kwargs):
        raise RuntimeError(
            f"{name} is a stub: HCU ships no sgl_kernel/flashinfer wheel"
        )

    return _raise


class _StubLoader(importlib.abc.Loader):
    # ModuleType directly: module_from_spec would re-enter this
    # create_module and recurse.
    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []

        def _ga(attr, _n=spec.name):
            if attr.startswith("__"):
                return None
            return _unavailable(f"{_n}.{attr}")

        module.__getattr__ = _ga
        return module

    def exec_module(self, module):
        pass


class _StubFinder:
    def find_spec(self, fullname, path=None, target=None):
        # scalar_type is optional in SGLang and has its own ImportError
        # fallback.  Pretending it exists changes scalar_types into a
        # generic stub function and breaks import-time attribute access
        # (uint4b8), so this finder must never fabricate it.
        if fullname == "sgl_kernel.scalar_type":
            return None
        if fullname.split(".")[0] not in _STUB_ROOTS:
            return None
        return importlib.util.spec_from_loader(fullname, _StubLoader())


if not any(type(f).__name__ == "_StubFinder" for f in sys.meta_path):
    sys.meta_path.append(_StubFinder())
