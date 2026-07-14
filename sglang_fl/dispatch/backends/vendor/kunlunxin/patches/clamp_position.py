"""Kunlunxin: route is_cuda-gated sglang JIT-CUDA plumbing kernels to native.

Several sglang scheduler/forward-batch utilities pick between a CUDA kernel and
a pure-torch fallback *at import time*, gated on ``is_cuda() or is_hip()``:

    if is_cuda() or is_hip():
        <name> = <name>_cuda        # JIT-compiled .cu via nvcc
    else:
        <name> = <name>_native      # pure torch

On Kunlunxin ``is_cuda()`` is True (cuda-compat device), so sglang selects the
CUDA branch. Those ``*_cuda`` variants JIT-compile a real ``.cu`` kernel with
the host nvcc using ``-std=c++20`` (see ``sglang/jit_kernel/utils.py``). Kunlun's
host toolchain is CUDA 11.7, whose nvcc caps at c++17, so compilation aborts:

    nvcc fatal : Value 'c++20' is not defined for option 'std'
    RuntimeError: ninja exited with status 1

Even if it compiled, the resulting NVIDIA SASS would not run on XPU.

Unlike the fused ops (silu_and_mul, rms_norm, rotary) these plumbing kernels are
*not* routed through FlagGems dispatch — they are called directly from the
scheduler / forward-batch code — so dispatch cannot intercept them. sglang ships
equivalent pure-torch implementations for non-cuda devices; we rebind each module
global to its native version. Every affected call site looks the global up at
call time, so rebinding the attribute takes effect regardless of import order.

Confirmed on the Qwen2.5 dense decode path:
  * overlap_utils._resolve_future_token_ids  (overlap scheduler)
  * forward_batch_info.clamp_position         (every ForwardBatch.init_new)

Fused ops (silu_and_mul, rms_norm, rotary) stay on FlagGems dispatch and are not
touched here.
"""

import logging

logger = logging.getLogger(__name__)
_applied = False


def patch_clamp_position():
    """Rebind clamp_position kernel to torch-native on klx."""
    global _applied
    if _applied:
        return

    from sglang.srt.managers import overlap_utils
    from sglang.srt.model_executor import forward_batch_info

    overlap_utils._resolve_future_token_ids = (
        overlap_utils._resolve_future_token_ids_native
    )
    forward_batch_info.clamp_position = forward_batch_info._clamp_position_native

    _applied = True
    logger.info(
        "patched JIT-CUDA plumbing kernels -> torch-native "
        "(resolve_future_token_ids, clamp_position; avoids nvcc -std=c++20 on Kunlunxin)"
    )
