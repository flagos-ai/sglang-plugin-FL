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

Also generates the `sgl_kernel_npu` import-face shim (Ascend). sglang 0.5.18
imports sgl_kernel_npu on the NPU branch across many processes — including
spawn scheduler workers at module-import time, before any plugin loads — so
the name must be importable as a real on-disk package, not aliased at
runtime. This wheel ships the same stub tree as the ascend E2E (authoritative
tree: 48 modules across 9 subpackages). Only ascend imports it; on every
other platform the is_npu branch never runs and the package is inert.

Run once before `pip wheel .`:

    python3 generate.py && pip wheel . --no-deps -w out
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "sgl_kernel")
NPU_PKG = os.path.join(HERE, "sgl_kernel_npu")

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


def _write_tree(path, pkgname, version, subpkgs, modules, prefix=""):
    """Write a stub package: init (docstring + __version__ + _Dummy +
    module-level __getattr__), a version submodule, empty subpackage inits and
    leaf modules that re-export the parent's module-level __getattr__."""
    os.makedirs(path, exist_ok=True)

    init = (
        f'"""FlagOS zero-{pkgname} shim for sglang 0.5.18 (import face only)."""\n'
        "\n"
        f'__version__ = "{version}"\n'
        "\n"
        + DUMMY.replace("sgl_kernel-stub", pkgname + "-stub")
        + "\n"
    )
    with open(os.path.join(path, "__init__.py"), "w") as f:
        f.write(init)

    with open(os.path.join(path, "version.py"), "w") as f:
        f.write(
            '"""Version submodule (import-face stub)."""\n'
            "\n"
            f'__version__ = "{version}"\n'
        )

    def reexport(dotted):
        # Re-export the __getattr__ of the enclosing package — for a nested
        # module that is sgl_kernel_npu.<subpkg>.<leaf>: the subpackage
        # sgl_kernel_npu.<subpkg> (importing it loads its __init__ which
        # re-exports the top-level __getattr__). For a bare module
        # (sgl_kernel_npu.kvcacheio): the top-level package.
        if "." in dotted:
            parent = pkgname + "." + dotted.rsplit(".", 1)[0]
        else:
            parent = pkgname
        return (
            f'"""Stub submodule `{dotted}` (zero-{pkgname} route)."""\n'
            "\n"
            f"from {parent} import __getattr__  # noqa: F401\n"
        )

    # Subpackage inits re-export the top-level __getattr__, so a bare
    # `import sgl_kernel_npu.norm` resolves to the real submodule and any
    # attribute access falls through to a _Dummy.
    for sub in subpkgs:
        d = os.path.join(path, *sub.split("."))
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "__init__.py"), "w") as f:
            f.write(
                f'"""Stub package `{pkgname}.{sub}` (zero-{pkgname} route)."""\n'
                "\n"
                f"from {pkgname} import __getattr__  # noqa: F401\n"
            )

    # Leaf modules: the sglang sites bind symbols via
    # `from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import <sym>`, so each
    # leaf must be a real importable module. Its only content re-exports the
    # parent __getattr__, exactly like the E2E-validated tree.
    for mod in modules:
        base, _, leaf = mod.rpartition(".")
        d = os.path.join(path, *base.split(".")) if base else path
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, leaf + ".py"), "w") as f:
            f.write(reexport(mod))

    print(
        f"generated {pkgname} shim package: {len(subpkgs)} subpackages, "
        f"{len(modules)} modules in {path}"
    )


# sgl_kernel_npu module layout from the ascend E2E-validated stub tree
# (48 leaf modules under 9 subpackages). Parent __init__s are implicit.
NPU_SUBPKGS = [
    "activation", "attention", "fla", "indexer", "kimi_k3",
    "mamba", "mem_cache", "norm", "sample",
]
NPU_MODULES = [
    "activation.situ", "activation.swiglu_oai", "activation.swiglu_oai_quant",
    "activation.swiglu_quant", "attention.fia_blockq_attention",
    "attention.gqa_share_sparse_attention", "attention.sinks_attention",
    "fla.chunk", "fla.fused_gdn_gating", "fla.fused_sigmoid_gating_recurrent",
    "fla.kda_chunk_delta_h", "fla.kda_gate", "fla.kda_prefill",
    "fla.kda_target_verify", "fla.layernorm_gated", "fla.solve_tril",
    "fla.utils", "indexer.flash_block_score_decode",
    "indexer.flash_block_score_prefill", "kimi_k3.attn_residual",
    "kvcacheio", "mamba.causal_conv1d", "mamba.causal_conv1d_verify",
    "mamba.mamba_state_update_triton", "mamba.speculative_state_scatter",
    "mem_cache.allocator", "mem_cache.kv_cache_store",
    "norm.add_rmsnorm_bias", "norm.fused_rope_qk_mqa",
    "norm.fused_split_qk_norm", "norm.l1_norm", "norm.rmsnorm_split",
    "norm.rmsnorm_without_weight", "norm.scale_shift",
    "norm.split_qkv_rmsnorm_rope",
    "norm.split_qkv_rmsnorm_rope_pos_cache_half_npu",
    "norm.split_qkv_tp_rmsnorm_rope", "sample.verify_tree_greedy",
]


def main() -> None:
    os.makedirs(PKG, exist_ok=True)

    init = (
        '"""FlagOS zero-sgl-kernel shim for sglang 0.5.18 (import face only)."""\n'
        "\n"
        '__version__ = "0.5.18"\n'
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
        '\n__version__ = "0.5.18"\n'
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

    _write_tree(
        NPU_PKG, "sgl_kernel_npu", "0.5.18",
        NPU_SUBPKGS, NPU_MODULES,
    )


if __name__ == "__main__":
    main()
