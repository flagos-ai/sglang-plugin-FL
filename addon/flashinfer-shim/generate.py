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

"""Generate the `flashinfer` import-face stub package for CUDA-alias vendors.

Some vendors' torch is CUDA-alias: `torch.version.cuda` is set, so sglang's
`is_cuda()` is True even though the platform is not NVIDIA and no flashinfer
build exists for it. Three module-level imports sit under a bare `if is_cuda`
with no availability check, and the first is fatal on every serve
(ServerArgs -> model_config -> quantization -> fp8 -> fp8_utils):

    from flashinfer import bmm_fp8 as _raw_bmm_fp8_batched
    ModuleNotFoundError: No module named 'flashinfer'

This has to be an on-disk package rather than a `sys.modules` seed installed
by the plugin: sglang's scheduler runs in a *spawned* worker which imports
`sglang.srt.managers.scheduler` (and through it the quantization package) at
module-import time — before any plugin loads. The same reasoning already made
`sgl_kernel_npu` an on-disk stub for ascend.

The allowlist is explicit: module-level `__getattr__` raises AttributeError for
anything else, so `hasattr(flashinfer_x, y)`-style capability probes elsewhere
in sglang keep answering honestly instead of taking a CUDA path. A symbol
outside the list fails at import exactly as it does today, which keeps the set
auditable.

Published to a vendor's index only where it is actually installed, and listed
in that backend's `deps_app` — a stub flashinfer must never shadow a real one.

Run once before `pip wheel .`:

    python3 generate.py && pip wheel . --no-deps -w out
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "flashinfer")

# Module -> symbols sglang imports from it at module level under `if is_cuda`.
ALLOWLIST = {
    "__init__": ("bmm_fp8",),
    "sampling": (
        "min_p_sampling_from_probs",
        "top_k_top_p_sampling_from_probs",
    ),
    "prefill": ("cudnn_batch_prefill_with_kv_cache",),
}

HEADER = '''\
"""Iluvatar/corex flashinfer stub — import face only, no kernels.

See addon/flashinfer-shim/README.md. Nothing here computes anything: corex has
no flashinfer build, so every stubbed call raises instead of silently
producing a wrong result.
"""

'''

TOP = HEADER + '''

def _unavailable(name):
    def _raise(*args, **kwargs):
        raise RuntimeError(
            f"{__name__}.{name} was called on a platform with no flashinfer "
            f"build — this call belongs behind is_flashinfer_available()"
        )

    _raise.__name__ = name
    return _raise


_ALLOWED = %(allowed)r


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
'''


def _submodule(name, symbols):
    return (
        HEADER
        + "\n"
        + "from flashinfer import __getattr__  # noqa: F401  (unknown names raise)\n"
        + "\n"
        + "\n".join(
            f"def {s}(*args, **kwargs):\n"
            f'    """Stub: corex has no flashinfer build."""\n'
            f"    raise RuntimeError(\n"
            f'        "{name}.{s} was called on a platform with no flashinfer "\n'
            f'        "build — this call belongs behind is_flashinfer_available()"\n'
            f"    )\n"
            for s in symbols
        )
    )


def main() -> None:
    os.makedirs(PKG, exist_ok=True)

    with open(os.path.join(PKG, "__init__.py"), "w") as f:
        f.write(TOP % {"allowed": ALLOWLIST["__init__"]})

    for mod, symbols in ALLOWLIST.items():
        if mod == "__init__":
            continue
        with open(os.path.join(PKG, mod + ".py"), "w") as f:
            f.write(_submodule(mod, symbols))

    print(f"generated flashinfer stub package: {len(ALLOWLIST)} modules in {PKG}")


if __name__ == "__main__":
    main()
