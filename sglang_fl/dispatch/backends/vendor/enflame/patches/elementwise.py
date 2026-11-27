"""GCU hook for sglang's ``fused_sigmoid_mul`` (Qwen3.5 gated attention).

SGLang imports the Triton implementation directly into the model module, so
there is no op class for the bridge layer to intercept; the symbol is hooked
instead.

Derived from SGLang v0.5.18; licensed under Apache-2.0.
triton_gcu miscompiles _fused_sigmoid_mul_kernel for the production strided
gate (the qkv chunk view) and for BLOCK_H=2048; a contiguous gate at tile
<= 1024 is bit-exact.  Keep the tile small and pack the gate before launch
(same house pattern as flashattention's q.contiguous()).  Kernel, launch
shape and semantics are unchanged.
"""

from __future__ import annotations

import torch
import triton

_BLOCK_H_MAX = 1024


def fused_sigmoid_mul_gcu(
    original, attn_output: torch.Tensor, gate: torch.Tensor, inplace: bool = False
) -> torch.Tensor:
    """``attn_output * sigmoid(gate)`` with a GCU-safe tile size."""
    from sglang.kernels.ops.elementwise.elementwise import _fused_sigmoid_mul_kernel

    if gate.ndim == 3 and attn_output.ndim == 2:
        # Strided gate: qkv was split/viewed, so gate rows are 2*head_dim apart.
        num_tokens, num_heads, head_dim = gate.shape
        hidden_dim = num_heads * head_dim
        assert attn_output.shape == (num_tokens, hidden_dim)
        gate_stride_row = gate.stride(0)
        gate_stride_head = gate.stride(1)
    else:
        assert (
            attn_output.shape == gate.shape
        ), "attn_output and gate must have the same shape"
        hidden_dim = attn_output.shape[-1]
        num_tokens = attn_output.numel() // hidden_dim
        head_dim = hidden_dim
        gate_stride_row = hidden_dim
        gate_stride_head = hidden_dim

    out = attn_output if inplace else torch.empty_like(attn_output)
    if hidden_dim <= _BLOCK_H_MAX:
        # One masked tail tile would have to be trusted; use the exact path.
        out.copy_(attn_output * torch.sigmoid(gate.reshape(num_tokens, hidden_dim)))
        return out

    block_h = 1024 if num_tokens < 1024 else 2048
    block_h = min(block_h, _BLOCK_H_MAX)
    # Pack the gate into a contiguous row; the kernel mis-addresses strided
    # gates on triton_gcu.
    gate = gate.reshape(num_tokens, hidden_dim)
    gate_stride_row = hidden_dim
    gate_stride_head = head_dim
    grid = (num_tokens, triton.cdiv(hidden_dim, block_h))
    _fused_sigmoid_mul_kernel[grid](
        out,
        attn_output,
        gate,
        gate_stride_row,
        gate_stride_head,
        hidden_dim,
        HEAD_DIM=head_dim,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return out


def patch_fused_sigmoid_mul():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.models.qwen3_5.fused_sigmoid_mul",
        fused_sigmoid_mul_gcu,
        HookType.AROUND,
    )
