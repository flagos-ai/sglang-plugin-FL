# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Make ``Qwen2MoeSparseMoeBlock.shared_expert_gate`` batch-invariant on NVIDIA.

Problem
-------
``shared_expert_gate`` is a plain ``torch.nn.Linear(hidden_size, 1)`` (N=1), so it
goes ``aten::linear -> aten::mm -> cuBLAS``. For N=1, cuBLAS picks its kernel by the
number of rows M (the total tokens in the batch): when M is a multiple of 8 it runs an
``nvjet`` GEMM kernel, otherwise a ``gemvx`` kernel. The two reduce over K in a
different order, so the same token can get a gate value that differs by one bf16 ulp
depending on which other requests share its prefill batch. The error is amplified
layer by layer and can flip the generated token (observed on Qwen3.6-35B-A3B, whose
MoE layers use this block).

Fix
---
Replace the gate's forward with a small Triton row-dot kernel whose tiling is fixed
(BLOCK_M, BLOCK_K, num_warps are constants, fp32 accumulation). Each output row is
reduced in the same order no matter how large M is, so the result no longer depends
on batch composition.

Scope: NVIDIA only, ``torch.nn.Linear`` gates with out_features == 1 and no bias.
Opt out with ``SGLANG_FL_BATCH_INVARIANT_SHARED_GATE=0``.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_patched = False

_BLOCK_M = 32
_BLOCK_K = 256
_NUM_WARPS = 4

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is not None:
        return _kernel

    import triton
    import triton.language as tl

    @triton.jit
    def _row_dot_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        M,
        K,
        stride_xm,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            cols = k0 + tl.arange(0, BLOCK_K)
            col_mask = cols < K
            x = tl.load(
                x_ptr + rows[:, None].to(tl.int64) * stride_xm + cols[None, :],
                mask=row_mask[:, None] & col_mask[None, :],
                other=0.0,
            )
            w = tl.load(w_ptr + cols, mask=col_mask, other=0.0)
            acc += tl.sum(x.to(tl.float32) * w.to(tl.float32)[None, :], axis=1)
        tl.store(y_ptr + rows, acc.to(y_ptr.dtype.element_ty), mask=row_mask)

    _kernel = _row_dot_kernel
    return _kernel


def batch_invariant_linear_n1(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``F.linear(x, weight)`` for ``weight`` of shape ``[1, K]``, batch-invariant.

    Every row of ``x`` is reduced with the same fixed tiling, so a row's output does
    not depend on how many other rows are in ``x``.
    """
    assert weight.dim() == 2 and weight.shape[0] == 1, weight.shape
    K = weight.shape[1]
    lead = x.shape[:-1]
    x2 = x.reshape(-1, K)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    w = weight.reshape(K)
    if not w.is_contiguous():
        w = w.contiguous()
    M = x2.shape[0]
    y = torch.empty((M,), device=x.device, dtype=torch.result_type(x2, w))
    if M > 0:
        grid = ((M + _BLOCK_M - 1) // _BLOCK_M,)
        _get_kernel()[grid](
            x2,
            w,
            y,
            M,
            K,
            x2.stride(0),
            BLOCK_M=_BLOCK_M,
            BLOCK_K=_BLOCK_K,
            num_warps=_NUM_WARPS,
        )
    return y.reshape(*lead, 1)


def _gate_forward(self: torch.nn.Linear, x: torch.Tensor) -> torch.Tensor:
    if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return batch_invariant_linear_n1(x, self.weight)
    return torch.nn.functional.linear(x, self.weight, self.bias)


def make_gate_batch_invariant(gate: object) -> bool:
    """Swap ``gate.forward`` for the batch-invariant kernel. Returns True if patched."""
    if (
        type(gate) is torch.nn.Linear
        and gate.out_features == 1
        and gate.bias is None
    ):
        gate.forward = _gate_forward.__get__(gate, torch.nn.Linear)
        return True
    return False


def patch_shared_expert_gate() -> None:
    """Patch ``Qwen2MoeSparseMoeBlock.__init__`` so new blocks get the invariant gate.

    Idempotent. No-op unless running on NVIDIA and the env switch is on.
    """
    global _patched
    if _patched:
        return

    flag = os.environ.get("SGLANG_FL_BATCH_INVARIANT_SHARED_GATE", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        logger.debug("shared_expert_gate patch disabled by env")
        return

    try:
        from sglang_fl.utils import get_device_info

        info = get_device_info()
        if info is None or info.vendor_name != "nvidia":
            logger.debug("shared_expert_gate patch: not NVIDIA, skipped")
            return

        from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

        orig_init = Qwen2MoeSparseMoeBlock.__init__

        def _patched_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            make_gate_batch_invariant(getattr(self, "shared_expert_gate", None))

        Qwen2MoeSparseMoeBlock.__init__ = _patched_init
        _patched = True
        logger.info("shared_expert_gate patched to batch-invariant N=1 kernel")
    except Exception as exc:
        logger.warning("shared_expert_gate patch failed: %s", exc)
