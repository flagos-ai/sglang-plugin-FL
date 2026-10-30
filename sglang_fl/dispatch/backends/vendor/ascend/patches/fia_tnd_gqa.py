"""Opt-in TND GQA for the verified B1/16K Q12/KV2/D256 prefill shape."""

from __future__ import annotations

import functools
import inspect
import logging
import os

import torch

logger = logging.getLogger(__name__)
_MARKER = "_sglang_fl_fia_tnd_gqa"


def _cpu_ints(value):
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                return None
            value = value.tolist()
        return [int(item) for item in value]
    except (TypeError, ValueError, RuntimeError):
        return None


def _eligible(self, q, k, v, layer, forward_batch):
    """Only select an unpadded, all-new EXTEND batch."""
    if os.getenv("SGLANG_FL_FIA_TND_GQA", "0") != "1":
        return None
    try:
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
    except ImportError:
        return None
    if forward_batch.forward_mode != ForwardMode.EXTEND:
        return None
    if not getattr(self, "use_fia", False) or getattr(self, "use_mla", False):
        return None
    cp_size = getattr(self, "attn_cp_size", 1)
    if getattr(self, "is_dllm_model", False) or (cp_size is not None and cp_size > 1):
        return None
    if getattr(self, "use_alibi", False):
        return None
    attn_type = getattr(layer, "attn_type", None)
    if getattr(attn_type, "name", None) == "ENCODER_ONLY":
        return None
    if getattr(layer, "sliding_window_size", -1) not in (-1, None):
        return None
    if getattr(layer, "logit_cap", 0) != 0:
        return None
    if q is None or k is None or v is None:
        return None
    if getattr(forward_batch, "encoder_lens", None) is not None:
        return None
    if getattr(forward_batch, "attn_cp_metadata", None) is not None:
        return None
    if getattr(layer, "is_cross_attention", False):
        return None

    # Exact CPU metadata excludes continuation, mixed and padded batches.
    if (
        getattr(forward_batch, "batch_size", None) != 1
        or _cpu_ints(getattr(forward_batch, "extend_seq_lens_cpu", None)) != [16384]
        or _cpu_ints(getattr(forward_batch, "seq_lens_cpu", None)) != [16384]
    ):
        return None
    prefix_lens = _cpu_ints(getattr(forward_batch, "extend_prefix_lens_cpu", None))
    if prefix_lens is not None and any(prefix_lens):
        return None

    try:
        q_heads = int(layer.tp_q_head_num)
        kv_heads = int(layer.tp_k_head_num)
        v_heads = int(getattr(layer, "tp_v_head_num", kv_heads))
        qk_dim = int(layer.qk_head_dim)
        v_dim = int(layer.v_head_dim)
    except (AttributeError, TypeError, RuntimeError, ValueError):
        return None
    # This is an operator-verified shape guard, not a model-name check.  It
    # prevents the 35B and other head layouts from silently taking this path.
    if (q_heads, kv_heads, v_heads, qk_dim, v_dim) != (12, 2, 2, 256, 256):
        return None
    if q.device.type != "npu" or not (q.device == k.device == v.device):
        return None
    if not (q.dtype == k.dtype == v.dtype == torch.bfloat16):
        return None
    if not (q.is_contiguous() and k.is_contiguous()):
        return None
    try:
        q_tnd = q.view(16384, 12, 256)
        k_tnd = k.view(16384, 2, 256)
        v_tnd = v.view(16384, 2, 256)
    except (RuntimeError, ValueError):
        return None
    # V is a zero-copy slice of packed [Q, gate, K, V]. FIA accepts the
    # verified strided layout; requiring a contiguous V excludes the model.
    if v_tnd.stride() not in ((512, 256, 1), (7168, 256, 1)):
        return None
    mask = getattr(self, "fia_mask", None)
    if (
        mask is None
        or tuple(mask.shape) != (2048, 2048)
        or mask.dtype != torch.bool
        or mask.device != q_tnd.device
        or not mask.is_contiguous()
    ):
        return None
    return q_tnd, k_tnd, v_tnd, mask


def patch_fia_tnd_gqa() -> bool:
    """Install the opt-in TND path while leaving graph/decode untouched."""
    if os.getenv("SGLANG_FL_FIA_TND_GQA", "0") != "1":
        return False
    try:
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
            AscendAttnBackend,
        )
    except (ImportError, RuntimeError):
        return False
    original = AscendAttnBackend.forward_extend
    if getattr(original, _MARKER, False):
        return False

    def wrapped(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        save_kv_cache=True,
        q_rope=None,
        k_rope=None,
        topk_indices=None,
        sinks=None,
        slopes=None,
    ):
        selected = None
        if save_kv_cache and all(
            value is None for value in (q_rope, k_rope, topk_indices, sinks, slopes)
        ):
            selected = _eligible(self, q, k, v, layer, forward_batch)
        if selected is None:
            return original(
                self,
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope=q_rope,
                k_rope=k_rope,
                topk_indices=topk_indices,
                sinks=sinks,
                slopes=slopes,
            )
        q_tnd, k_tnd, v_tnd, mask = selected
        # Keep the stock cache visibility/order: KV is committed before the
        # attention result is returned to the layer.
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer, forward_batch.out_cache_loc, k, v
        )
        # Never hide an operator/OOM/device failure by retrying after KV writes.
        attn_output = torch.ops.npu.npu_fused_infer_attention_score(
            q_tnd,
            k_tnd,
            v_tnd,
            num_heads=12,
            num_key_value_heads=2,
            input_layout="TND",
            atten_mask=mask,
            sparse_mode=3,
            scale=layer.scaling,
            next_tokens=0,
            actual_seq_lengths=[16384],
            actual_seq_lengths_kv=[16384],
        )[0]
        if tuple(attn_output.shape) != (16384, 12, 256) or attn_output.dtype != q.dtype:
            raise RuntimeError(
                f"unexpected TND FIA output {tuple(attn_output.shape)} {attn_output.dtype}"
            )
        if not getattr(self, "_sglang_fl_fia_tnd_hit_logged", False):
            logger.info("FIA_TND_FAST_HIT: tokens=16384 q_heads=12 kv_heads=2")
            self._sglang_fl_fia_tnd_hit_logged = True
        return attn_output.contiguous().view(16384, 3072)

    # Check before @wraps can copy the original's __wrapped__/__signature__.
    try:
        compatible = tuple(inspect.signature(original).parameters) == tuple(
            inspect.signature(wrapped).parameters
        )
    except (TypeError, ValueError):
        compatible = False
    if not compatible:
        logger.warning("Skipping FIA TND patch: incompatible forward_extend signature")
        return False
    wrapped = functools.wraps(original)(wrapped)
    setattr(wrapped, _MARKER, True)
    AscendAttnBackend.forward_extend = wrapped
    logger.info("Installed opt-in FIA TND+GQA fresh-prefill path")
    return True
