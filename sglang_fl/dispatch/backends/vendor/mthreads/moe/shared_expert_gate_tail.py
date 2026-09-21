# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in MUSA shared-expert gate-tail fusion.

The plugin's guarded shared-expert hook integrates this default-off kernel
for the small, bias-free shared-expert gate used by the
Qwen3.6 product shape (M=4, H=2048):

    F.linear(hidden, gate_weight)
      -> BF16 materialization -> sigmoid
      -> BF16 materialization -> broadcast multiply(shared_output)
      -> BF16 output materialization

The kernel keeps the two BF16 conversion boundaries in registers.  That is
the numerical contract to be checked against the actual MUSA eager path; it
does not claim to reproduce the vendor GEMV reduction order.  If that order
differs, the caller must keep the original path (or evaluate a narrower
sigmoid+multiply-only candidate) rather than relax a bitwise comparator.

The implementation intentionally avoids allocations, host synchronization,
or host-side scalar extraction on the candidate path.  A caller that is in a
graph capture must provide a pointer-stable ``out`` buffer.  There is no model
call-site integration in this file itself.  Correctness-only callers may provide
preallocated ``gate_logits_out`` and ``gate_prob_out`` buffers; ordinary
production calls leave both unset, so no debug stores are emitted.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:  # Importing this module must remain possible in CPU-only tests.
    import triton
    import triton.language as tl
except (ImportError, ModuleNotFoundError):  # pragma: no cover - runtime image.
    triton = None
    tl = None


FUSE_ENV = "SGLANG_MUSA_SHARED_EXPERT_GATE_TAIL_FUSED"
PRODUCT_M = 4
PRODUCT_H = 2048
BLOCK_H = 256


def _env_enabled() -> bool:
    return os.environ.get(FUSE_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _capture_active(device: torch.device) -> bool:
    """Return capture state without adding a sync or a device allocation."""

    if device.type != "musa":
        return False
    try:
        # Prefer the backend-named module.  Some torch_musa builds expose
        # capture state through ``torch.musa`` while others proxy the CUDA
        # namespace; neither behavior is assumed as a hardware fact.
        for module_name in (device.type, "cuda"):
            device_module = getattr(torch, module_name, None)
            capture_fn = getattr(device_module, "is_current_stream_capturing", None)
            if capture_fn is not None:
                return bool(capture_fn())
        return True
    except (
        AssertionError,
        AttributeError,
        NotImplementedError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        # A missing capture query is not permission to allocate in a graph.
        # The caller can pass ``capturing=True`` to the guard when required.
        return True


def _storage_span(tensor: torch.Tensor) -> tuple[int, int] | None:
    """Return a conservative storage interval for fail-closed alias checks."""

    try:
        storage = tensor.untyped_storage()
        start = int(tensor.data_ptr())
        return start, start + int(storage.nbytes())
    except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError):
        return None


def _may_alias(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Conservatively detect overlap, including views sharing one storage."""

    if left.device != right.device:
        return False
    left_span = _storage_span(left)
    right_span = _storage_span(right)
    if left_span is None or right_span is None:
        return True
    return max(left_span[0], right_span[0]) < min(left_span[1], right_span[1])


def candidate_guard_reason(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    gate_logits_out: torch.Tensor | None = None,
    gate_prob_out: torch.Tensor | None = None,
    enabled: bool | None = None,
    capturing: bool | None = None,
) -> str:
    """Explain why the candidate is or is not eligible.

    The reasons are intentionally conservative and stable enough for a
    harness to record.  ``enabled`` is injectable for CPU policy tests; the
    production default is the environment variable above, whose default is
    disabled.
    """

    if enabled is None:
        enabled = _env_enabled()
    if not enabled:
        return "disabled"

    if not all(
        isinstance(tensor, torch.Tensor)
        for tensor in (hidden, gate_weight, shared_output)
    ):
        return "non_tensor_input"
    if hidden.ndim != 2 or shared_output.ndim != 2 or gate_weight.ndim != 2:
        return "rank_not_2"
    if hidden.shape != (PRODUCT_M, PRODUCT_H):
        return "hidden_shape_not_M4_H2048"
    if gate_weight.shape != (1, PRODUCT_H):
        return "gate_weight_shape_not_1xH"
    if shared_output.shape != (PRODUCT_M, PRODUCT_H):
        return "shared_output_shape_not_M4_H2048"
    if hidden.dtype != torch.bfloat16 or gate_weight.dtype != torch.bfloat16:
        return "input_dtype_not_bfloat16"
    if shared_output.dtype != torch.bfloat16:
        return "shared_dtype_not_bfloat16"
    if hidden.device != gate_weight.device or hidden.device != shared_output.device:
        return "device_mismatch"
    if (
        not hidden.is_contiguous()
        or not gate_weight.is_contiguous()
        or not shared_output.is_contiguous()
    ):
        return "input_layout_not_contiguous"
    if tuple(hidden.stride()) != (PRODUCT_H, 1):
        return "hidden_stride_not_MajorH"
    if tuple(gate_weight.stride()) != (PRODUCT_H, 1):
        return "gate_stride_not_H_major"
    if tuple(shared_output.stride()) != (PRODUCT_H, 1):
        return "shared_stride_not_MajorH"
    if _may_alias(hidden, gate_weight):
        return "hidden_aliases_gate_weight"
    if _may_alias(hidden, shared_output):
        return "hidden_aliases_shared_output"
    if _may_alias(gate_weight, shared_output):
        return "gate_weight_aliases_shared_output"

    if out is not None:
        if not isinstance(out, torch.Tensor):
            return "output_not_tensor"
        if out.shape != (PRODUCT_M, PRODUCT_H):
            return "output_shape_not_M4_H2048"
        if out.dtype != torch.bfloat16:
            return "output_dtype_not_bfloat16"
        if out.device != hidden.device:
            return "output_device_mismatch"
        if not out.is_contiguous() or tuple(out.stride()) != (PRODUCT_H, 1):
            return "output_layout_not_contiguous"
        if any(
            _may_alias(out, tensor) for tensor in (hidden, gate_weight, shared_output)
        ):
            return "output_aliases_input"
    elif capturing is True:
        return "output_required_during_capture"

    debug_outputs = (
        ("gate_logits_debug", gate_logits_out),
        ("gate_prob_debug", gate_prob_out),
    )
    for debug_name, debug_output in debug_outputs:
        if debug_output is None:
            continue
        if not isinstance(debug_output, torch.Tensor):
            return f"{debug_name}_not_tensor"
        if debug_output.shape != (PRODUCT_M, 1):
            return f"{debug_name}_shape_not_Mx1"
        if debug_output.dtype != torch.bfloat16:
            return f"{debug_name}_dtype_not_bfloat16"
        if debug_output.device != hidden.device:
            return f"{debug_name}_device_mismatch"
        if not debug_output.is_contiguous() or tuple(debug_output.stride()) != (1, 1):
            return f"{debug_name}_layout_not_contiguous"
        if any(
            _may_alias(debug_output, tensor)
            for tensor in (hidden, gate_weight, shared_output, out)
            if tensor is not None
        ):
            return f"{debug_name}_aliases_input_or_output"
    if (
        gate_logits_out is not None
        and gate_prob_out is not None
        and _may_alias(gate_logits_out, gate_prob_out)
    ):
        return "debug_outputs_alias"

    if hidden.device.type != "musa":
        return "device_not_musa"
    return "eligible"


def materialized_reference(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    gate_logits_out: torch.Tensor | None = None,
    gate_prob_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference with the production BF16 materialization boundaries.

    The explicit ``to(torch.bfloat16)`` operations model the required store
    boundaries even when the backend already returns BF16.  This helper is
    for a correctness oracle and for safe fallback in this standalone
    candidate; a model integration should call its existing original chain
    when the guard is not eligible.  The optional debug buffers are written
    only to make the first bitwise mismatch observable during correctness
    work; they are not part of the product output.
    """

    if hidden.shape != (PRODUCT_M, PRODUCT_H):
        raise ValueError("reference expects M=4,H=2048 hidden states")
    if gate_weight.shape != (1, PRODUCT_H):
        raise ValueError("reference expects a [1,2048] gate weight")
    if shared_output.shape != (PRODUCT_M, PRODUCT_H):
        raise ValueError("reference expects M=4,H=2048 shared output")
    if hidden.dtype != torch.bfloat16 or gate_weight.dtype != torch.bfloat16:
        raise ValueError("reference expects BF16 hidden and gate weight")
    if shared_output.dtype != torch.bfloat16:
        raise ValueError("reference expects BF16 shared output")
    if (
        not hidden.is_contiguous()
        or not gate_weight.is_contiguous()
        or not shared_output.is_contiguous()
    ):
        raise ValueError("reference expects contiguous inputs")

    if out is None:
        out = torch.empty_like(shared_output)
    else:
        reason = candidate_guard_reason(
            hidden,
            gate_weight,
            shared_output,
            out,
            enabled=True,
            capturing=False,
        )
        if reason == "output_aliases_input":
            raise ValueError("reference output must not alias an input")
        if out.shape != shared_output.shape or out.dtype != torch.bfloat16:
            raise ValueError("reference output has an invalid shape or dtype")

    debug_reason = candidate_guard_reason(
        hidden,
        gate_weight,
        shared_output,
        out,
        gate_logits_out=gate_logits_out,
        gate_prob_out=gate_prob_out,
        enabled=True,
        capturing=False,
    )
    if debug_reason not in {"device_not_musa", "eligible"}:
        raise ValueError(f"invalid debug/reference buffers: {debug_reason}")

    # Each assignment is a semantic checkpoint.  Do not collapse these
    # stages while establishing a comparator for the production path.
    gate_logits = F.linear(hidden, gate_weight, bias=None)
    gate_logits_bf16 = gate_logits.to(torch.bfloat16)
    if gate_logits_out is not None:
        gate_logits_out.copy_(gate_logits_bf16)
    gate_prob = torch.sigmoid(gate_logits_bf16)
    gate_prob_bf16 = gate_prob.to(torch.bfloat16)
    if gate_prob_out is not None:
        gate_prob_out.copy_(gate_prob_bf16)
    product = gate_prob_bf16 * shared_output
    out.copy_(product.to(torch.bfloat16))
    return out


if triton is not None:

    @triton.jit
    def _shared_expert_gate_tail_kernel(
        hidden_ptr,
        gate_ptr,
        shared_ptr,
        out_ptr,
        gate_logits_out_ptr,
        gate_prob_out_ptr,
        hidden_stride_m,
        hidden_stride_h,
        shared_stride_m,
        shared_stride_h,
        out_stride_m,
        out_stride_h,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
        WRITE_DEBUG_LOGITS: tl.constexpr,
        WRITE_DEBUG_PROB: tl.constexpr,
    ):
        """One row program; deliberately no temporary gate/probability tensor."""

        row = tl.program_id(axis=0)
        offsets = tl.arange(0, BLOCK_H)
        accumulator = tl.zeros((BLOCK_H,), dtype=tl.float32)

        # This is an explicit FP32 accumulation, not a claim about the
        # reduction order used by MUSA F.linear.  Bitwise validation must
        # compare the resulting BF16 gate logits before accepting this path.
        for start in range(0, H, BLOCK_H):
            columns = start + offsets
            mask = columns < H
            x = tl.load(
                hidden_ptr + row * hidden_stride_m + columns * hidden_stride_h,
                mask=mask,
                other=0,
            )
            weight = tl.load(gate_ptr + columns, mask=mask, other=0)
            accumulator += x.to(tl.float32) * weight.to(tl.float32)

        # Preserve the first eager materialization boundary before sigmoid.
        gate_logits_bf16 = tl.sum(accumulator, axis=0).to(tl.bfloat16)
        if WRITE_DEBUG_LOGITS:
            tl.store(gate_logits_out_ptr + row, gate_logits_bf16)
        gate_prob_bf16 = tl.sigmoid(gate_logits_bf16.to(tl.float32)).to(tl.bfloat16)
        if WRITE_DEBUG_PROB:
            tl.store(gate_prob_out_ptr + row, gate_prob_bf16)

        # Preserve a BF16 broadcast multiply and the final BF16 output store.
        for start in range(0, H, BLOCK_H):
            columns = start + offsets
            mask = columns < H
            shared = tl.load(
                shared_ptr + row * shared_stride_m + columns * shared_stride_h,
                mask=mask,
                other=0,
            )
            scaled = shared * gate_prob_bf16
            tl.store(
                out_ptr + row * out_stride_m + columns * out_stride_h,
                scaled,
                mask=mask,
            )


def fused_shared_expert_gate_tail(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    gate_logits_out: torch.Tensor | None = None,
    gate_prob_out: torch.Tensor | None = None,
    enabled: bool | None = None,
) -> torch.Tensor:
    """Run the opt-in candidate or return the materialized reference.

    ``out`` is optional only for eager mode.  During capture, a missing
    output is a hard guard failure rather than an allocation.  All rejected
    shapes and devices use the reference here so the standalone harness is
    useful on CPU; a production integration should retain its original model
    call for those reasons.  ``gate_logits_out`` and ``gate_prob_out`` are
    correctness-only, caller-owned `[4,1]` BF16 buffers.
    """

    capturing = _capture_active(hidden.device)
    reason = candidate_guard_reason(
        hidden,
        gate_weight,
        shared_output,
        out,
        gate_logits_out=gate_logits_out,
        gate_prob_out=gate_prob_out,
        enabled=enabled,
        capturing=capturing,
    )
    if reason != "eligible":
        if reason == "output_required_during_capture":
            raise RuntimeError(
                "shared gate-tail candidate requires a preallocated output in capture"
            )
        return materialized_reference(
            hidden,
            gate_weight,
            shared_output,
            out,
            gate_logits_out=gate_logits_out,
            gate_prob_out=gate_prob_out,
        )

    if triton is None:  # pragma: no cover - runtime-image dependent.
        return materialized_reference(
            hidden,
            gate_weight,
            shared_output,
            out,
            gate_logits_out=gate_logits_out,
            gate_prob_out=gate_prob_out,
        )
    if out is None:
        out = torch.empty_like(shared_output)
    debug_logits_ptr = gate_logits_out if gate_logits_out is not None else out
    debug_prob_ptr = gate_prob_out if gate_prob_out is not None else out

    _shared_expert_gate_tail_kernel[(PRODUCT_M,)](
        hidden,
        gate_weight,
        shared_output,
        out,
        debug_logits_ptr,
        debug_prob_ptr,
        hidden.stride(0),
        hidden.stride(1),
        shared_output.stride(0),
        shared_output.stride(1),
        out.stride(0),
        out.stride(1),
        H=PRODUCT_H,
        BLOCK_H=BLOCK_H,
        WRITE_DEBUG_LOGITS=gate_logits_out is not None,
        WRITE_DEBUG_PROB=gate_prob_out is not None,
    )
    return out


__all__ = [
    "BLOCK_H",
    "FUSE_ENV",
    "PRODUCT_H",
    "PRODUCT_M",
    "candidate_guard_reason",
    "fused_shared_expert_gate_tail",
    "materialized_reference",
]
