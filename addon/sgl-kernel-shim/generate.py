#!/usr/bin/env python3
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

"""Generate the `sgl_kernel` zero-kernel shim package for sglang 0.5.18.

Satisfies the 0.5.18 import face (82 `from sgl_kernel import <sym>` sites +
29 submodule imports) with stubs — runtime symbols are never called on the
flagos path (ops go through flag_gems). The 0.4.x `sglang_kernel` stub wheels
satisfy only the pip distribution `sglang-kernel`, NOT the `sgl_kernel` module
0.5.18 imports; this package closes that gap.

Run once before `pip wheel .`:

    python3 generate.py && pip wheel . --no-deps -w out
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "sgl_kernel")

# Submodules 0.5.18 imports (from `grep "from sgl_kernel\.[a-z_]* import"`).
SUBMODULES = [
    "allreduce", "attention", "cutlass_moe", "debug_utils", "elementwise",
    "expert_specialization", "flash_attn", "flash_mla", "gemm", "grammar",
    "infllm_v", "kvcacheio", "load_utils", "mamba", "memory", "metal", "moe",
    "musa", "quantization", "sampling", "scalar_type", "sparse_flash_attn",
    "spatial", "speculative", "test_utils", "testing", "top_k", "utils",
]

DUMMY = '''\
class _Dummy:
    """Universal stand-in: any attribute, call, or index returns a _Dummy.

    Import-time code never calls through (flagos backend ops live in
    flag_gems), so the shim only has to survive attribute access and
    `from sgl_kernel.x import y`. Making it callable/iterable is belt-and-
    braces for a stray runtime reference on an unguarded path.
    """

    _name = "<sgl_kernel-stub>"

    def __init__(self, name="<sgl_kernel-stub>"):
        self._name = name

    def __call__(self, *a, **k):
        return _Dummy(self._name + "()")

    def __getattr__(self, n):
        return _Dummy(self._name + "." + n)

    def __getitem__(self, k):
        return _Dummy(self._name + "[%r]" % (k,))

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __bool__(self):
        return False

    def __int__(self):
        return 0

    def __float__(self):
        return 0.0

    def __str__(self):
        return self._name

    def __repr__(self):
        return self._name

    def __eq__(self, other):
        return isinstance(other, _Dummy)

    def __hash__(self):
        return 0

    def __add__(self, o):
        return _Dummy()
    __radd__ = __add__

    def __sub__(self, o):
        return _Dummy()
    __rsub__ = __sub__

    def __mul__(self, o):
        return _Dummy()
    __rmul__ = __mul__

    def __truediv__(self, o):
        return _Dummy()
    __rtruediv__ = __truediv__

    def __neg__(self):
        return _Dummy()

    def __pos__(self):
        return _Dummy()

    def __abs__(self):
        return _Dummy()


def __getattr__(name):
    return _Dummy("sgl_kernel." + name)
'''


def main() -> None:
    os.makedirs(PKG, exist_ok=True)

    init = (
        '"""FlagOS zero-sgl-kernel shim for sglang 0.5.18 (import face only)."""\n'
        "\n"
        '__version__ = "0.5.18+flagos-shim"\n'
        "\n"
        + DUMMY
        + "\n"
    )
    with open(os.path.join(PKG, "__init__.py"), "w") as f:
        f.write(init)

    version_py = (
        '"""Version submodule — `sglang/kernels/aot/python/sgl_kernel/__init__.py`\n'
        'does `from sgl_kernel.version import __version__`.\n'
        '"""\n'
        '\n__version__ = "0.5.18+flagos-shim"\n'
    )
    with open(os.path.join(PKG, "version.py"), "w") as f:
        f.write(version_py)

    for sub in SUBMODULES:
        if sub == "version":
            continue
        body = (
            f'"""Stub submodule `sgl_kernel.{sub}` (zero-sgl-kernel route)."""\n'
            "\n"
            "from sgl_kernel import __getattr__  # noqa: F401\n"
        )
        with open(os.path.join(PKG, sub + ".py"), "w") as f:
            f.write(body)

    print(f"generated sgl_kernel shim package: {len(SUBMODULES)} submodules in {PKG}")


if __name__ == "__main__":
    main()
