# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Import-only stand-ins for vendor kernel packages that do not support this GPU.

sglang imports aiter and sgl_kernel at module level on ROCm. On GPUs these
packages do not support (e.g. gfx1201), every name resolves to a placeholder
that raises on use, so an unexpected kernel call fails loudly instead of
silently computing nothing.
"""

import importlib.abc
import importlib.machinery
import sys
import types


class _Missing:
    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Missing(f"{self._name}.{attr}")

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f"stub: {self._name} is unavailable on this GPU")

    def __repr__(self):
        return f"<stub {self._name}>"


def _module_getattr(module_name):
    def __getattr__(attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Missing(f"{module_name}.{attr}")

    return __getattr__


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, prefix):
        self._prefix = prefix + "."

    def find_spec(self, fullname, path, target=None):
        if fullname.startswith(self._prefix):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []
        module.__getattr__ = _module_getattr(spec.name)
        return module

    def exec_module(self, module):
        pass


def install(package):
    sys.meta_path.insert(0, _StubFinder(package))
    return _module_getattr(package)
