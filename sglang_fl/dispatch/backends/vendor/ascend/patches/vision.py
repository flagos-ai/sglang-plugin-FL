"""SGLang v0.5.18 multimodal-attention compatibility for CANN 8.5."""

from __future__ import annotations

import inspect
from functools import wraps

import torch


_PATCH_MARKER = "_sglang_fl_unaligned_head_fallback"
_CANN_8_5_FUSED_HEAD_SIZES = frozenset({64, 128, 192})
_QWEN3_6_HEAD_SIZE = 72
_QWEN3_6_PADDED_HEAD_SIZE = 128
_EXPECTED_FORWARD_PARAMETERS = (
    "self",
    "q",
    "k",
    "v",
    "cu_seqlens",
    "bsz",
    "seq_len",
    "softmax_scale",
    "forward_metadata",
    "attention_mask",
    "kwargs",
)


def patch_vision_ascend_attention() -> None:
    """Route unsupported NPU vision head dimensions through existing SDPA.

    CANN 8.5's ``npu_fused_infer_attention_score`` accepts BF16/FP16 head
    sizes 64, 128, and 192. SGLang v0.5.18 calls that operator directly from
    ``VisionAscendAttention``; Qwen3.6-VL has head size 72 and fails before its
    first image response. Zero-pad that known shape to 128 while retaining the
    original 1/sqrt(72) scale, then slice the fused result back to 72. This is
    mathematically equivalent to the unpadded attention and preserves the
    fused path. Reuse the upstream class's configured SDPA fallback for any
    other unsupported head size.
    """

    from sglang.srt.layers.attention.vision import VisionAscendAttention

    original_forward = VisionAscendAttention.forward
    if getattr(original_forward, _PATCH_MARKER, False):
        return

    actual_parameters = tuple(inspect.signature(original_forward).parameters)
    if actual_parameters != _EXPECTED_FORWARD_PARAMETERS:
        raise RuntimeError(
            "Unsupported SGLang VisionAscendAttention.forward interface for "
            "the Ascend CANN 8.5 compatibility patch: "
            f"{actual_parameters}, expected {_EXPECTED_FORWARD_PARAMETERS}."
        )

    @wraps(original_forward)
    def forward(
        self,
        q,
        k,
        v,
        cu_seqlens,
        bsz,
        seq_len,
        softmax_scale=None,
        forward_metadata=None,
        attention_mask=None,
        **kwargs,
    ):
        head_size = q.shape[-1]
        if (
            head_size == _QWEN3_6_HEAD_SIZE
            and k.shape[-1] == head_size
            and v.shape[-1] == head_size
            and attention_mask is None
        ):
            padding = _QWEN3_6_PADDED_HEAD_SIZE - head_size
            q_padded = torch.nn.functional.pad(q, (0, padding))
            k_padded = torch.nn.functional.pad(k, (0, padding))
            v_padded = torch.nn.functional.pad(v, (0, padding))
            scale = head_size**-0.5 if softmax_scale is None else softmax_scale
            output = original_forward(
                self,
                q_padded,
                k_padded,
                v_padded,
                cu_seqlens,
                bsz,
                seq_len,
                softmax_scale=scale,
                forward_metadata=forward_metadata,
                attention_mask=attention_mask,
                **kwargs,
            )
            return output[..., :head_size].contiguous()
        if head_size not in _CANN_8_5_FUSED_HEAD_SIZES:
            return self.sdpa_fallback(
                q=q,
                k=k,
                v=v,
                cu_seqlens=cu_seqlens,
                bsz=bsz,
                seq_len=seq_len,
                softmax_scale=softmax_scale,
                forward_metadata=forward_metadata,
                attention_mask=attention_mask,
                **kwargs,
            )
        return original_forward(
            self,
            q,
            k,
            v,
            cu_seqlens,
            bsz,
            seq_len,
            softmax_scale=softmax_scale,
            forward_metadata=forward_metadata,
            attention_mask=attention_mask,
            **kwargs,
        )

    setattr(forward, _PATCH_MARKER, True)
    VisionAscendAttention.forward = forward
