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

"""Iluvatar: give vendor triton the PDL intrinsics its source references.

sglang's attention kernels call `tl.extra.cuda.gdc_wait()` /
`gdc_launch_dependents()` under a `USE_PDL` constexpr. Those are part of
triton's extra.cuda namespace from 3.3 on; corex's vendor triton is 3.2, which
does not define them — and its JIT resolves *every* attribute in the function
body during an AST pre-pass, so the reference aborts compilation even though
the branch is not taken:

    File "/opt/triton/triton/runtime/jit.py", line 307, in visit_Attribute
      ret = getattr(lhs, node.attr)
    AttributeError: module 'triton.language.extra.cuda' has no attribute
                    'gdc_launch_dependents'

With USE_PDL false (is_arch_support_pdl() answers False — PDL is NVIDIA
Hopper+) the calls never execute, so a no-op satisfies the pre-pass without
changing behaviour. Verified on the node: a kernel carrying the same guarded
reference compiles and runs once the attributes exist.

Only patched when triton actually lacks them, so a triton that ships the real
intrinsics is left alone.
"""

import logging

logger = logging.getLogger(__name__)
_applied = False

_MISSING = ("gdc_wait", "gdc_launch_dependents")


def _noop(*args, **kwargs):
    return None


# triton's dependency pass only accepts a call it recognises as triton-side:
# `func.__module__.startswith("triton")`. Vendor triton 3.1.0 (corex 4.4.0)
# *asserts* on that during the AST pre-pass and aborts the compile:
#
#     AssertionError: Function "_noop" is being called from a Triton function
#     but is not a Triton function itself. Decorate it with @triton.jit
#
# 3.2.0 (corex 4.5.0) runs the same test without asserting, which is why the
# plain function was enough there. The function is installed *into*
# triton.language.extra.cuda, so answering for that namespace is truthful.
_noop.__module__ = "triton.language.extra.cuda"


def patch_triton_pdl_intrinsics():
    """Add no-op PDL intrinsics to triton.language.extra.cuda when absent."""
    global _applied
    if _applied:
        return
    _applied = True

    try:
        import triton.language.extra.cuda as tlc
    except ImportError:
        logger.info("iluvatar: triton.language.extra.cuda absent, nothing to patch")
        return

    added = []
    for name in _MISSING:
        if not hasattr(tlc, name):
            setattr(tlc, name, _noop)
            added.append(name)

    if added:
        logger.info(
            "iluvatar: added no-op triton PDL intrinsics %s (vendor triton lacks "
            "them; unused because USE_PDL is false)",
            added,
        )
