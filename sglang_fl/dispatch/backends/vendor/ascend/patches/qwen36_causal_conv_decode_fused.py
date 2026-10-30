# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""BF16 width-four causal-convolution decode with explicit rounding points."""

import functools
import inspect
import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)
_MARKER = "_sglang_fl_fused_conv_decode"
_SHAPES = frozenset({(4096, 4), (5120, 4)})


@triton.jit
def _bf16_round_f32(value):
    # Current Ascend lowering does not reliably retain an immediate
    # float32->bfloat16->float32 round trip.  Integer RNE makes the required
    # rounding point explicit while retaining float32 for subsequent math.
    bits = value.to(tl.uint32, bitcast=True)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    special = (bits & 0x7F800000) == 0x7F800000
    return tl.where(special, bits, rounded).to(tl.float32, bitcast=True)


@triton.jit
def _conv_update_bf16_four_kernel(
    X,
    SELECTED,
    STATES,
    W,
    INDICES,
    OUT,
    C: tl.constexpr,
    SLOTS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    COALESCED_WEIGHT: tl.constexpr,
):
    # Snapshot selected slots before launching, so duplicate padding indices
    # never read a state concurrently modified by another request program.
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    active = cols < C
    idx = tl.load(INDICES + row)
    idx = tl.where(idx < 0, idx + SLOTS, idx)
    state_off = idx * (3 * C) + cols
    selected_off = row * (3 * C) + cols
    x_off = row * C + cols
    h0 = tl.load(SELECTED + selected_off, active, other=0).to(tl.float32)
    h1_bf16 = tl.load(SELECTED + selected_off + C, active, other=0)
    h2_bf16 = tl.load(SELECTED + selected_off + 2 * C, active, other=0)
    x_bf16 = tl.load(X + x_off, active, other=0)
    h1 = h1_bf16.to(tl.float32)
    h2 = h2_bf16.to(tl.float32)
    x = x_bf16.to(tl.float32)
    if COALESCED_WEIGHT:
        packed = tl.load(
            W + cols[:, None] * 4 + tl.arange(0, 4)[None, :],
            active[:, None],
            other=0,
        ).to(tl.float32)
        even, odd = tl.split(tl.reshape(packed, (BLOCK_C, 2, 2)))
        w0, w2 = tl.split(even)
        w1, w3 = tl.split(odd)
    else:
        w0 = tl.load(W + cols, active, other=0).to(tl.float32)
        w1 = tl.load(W + cols + C, active, other=0).to(tl.float32)
        w2 = tl.load(W + cols + 2 * C, active, other=0).to(tl.float32)
        w3 = tl.load(W + cols + 3 * C, active, other=0).to(tl.float32)
    # Match torch BF16 mul -> native sum -> BF16 SiLU.  Omitting either
    # BF16 conversion changes the reference arithmetic and is not allowed.
    p0 = _bf16_round_f32(h0 * w0)
    p1 = _bf16_round_f32(h1 * w1)
    p2 = _bf16_round_f32(h2 * w2)
    p3 = _bf16_round_f32(x * w3)
    total = _bf16_round_f32((p0 + p1) + (p2 + p3))
    out = tl.fdiv(total, 1.0 + tl.exp(-total))
    tl.store(OUT + x_off, out, active)
    tl.store(STATES + state_off, h1_bf16, active)
    tl.store(STATES + state_off + C, h2_bf16, active)
    tl.store(STATES + state_off + 2 * C, x_bf16, active)


def fused_decode_conv(x, states, weight, indices, coalesced_weight=False):
    """Production decode path with optional coalesced four-tap weight loads."""
    batch, channels = x.shape
    output = torch.empty_like(x)
    selected = states[indices]
    block_channels = 2048
    _conv_update_bf16_four_kernel[(batch, triton.cdiv(channels, block_channels))](
        x,
        selected,
        states,
        weight,
        indices,
        output,
        C=channels,
        SLOTS=states.shape[0],
        BLOCK_C=block_channels,
        COALESCED_WEIGHT=coalesced_weight,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return output


def _transposed_weight(layer):
    # Keep this copy inside the captured graph.  SGLang supports in-place
    # weight updates without recapture, so a Python identity/version cache
    # would retain stale values during graph replay.
    return layer.conv_weights.transpose(0, 1).contiguous()


def _supported(x, states, layer, forward_batch, indices):
    weight = layer.conv_weights
    if not (
        isinstance(x, torch.Tensor)
        and x.device.type == "npu"
        and not torch.is_grad_enabled()
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and 0 < x.shape[0] <= 64
        and x.shape[0] == forward_batch.batch_size
        and x.is_contiguous()
        and tuple(weight.shape) in _SHAPES
        and weight.shape[0] == x.shape[1]
        and weight.dtype == x.dtype
        and weight.is_contiguous()
        and states.dtype == x.dtype
        and states.device == x.device == weight.device
        and states.ndim == 3
        and tuple(states.shape[1:]) == (3, x.shape[1])
        and states.is_contiguous()
        and indices.device == x.device
        and indices.ndim == 1
        and indices.numel() == x.shape[0]
        and indices.is_contiguous()
        and indices.dtype in (torch.int32, torch.int64)
        and layer.bias is None
        and layer.activation in ("silu", "swish")
        and forward_batch.forward_mode.is_decode()
        and forward_batch.spec_algorithm.is_none()
    ):
        return False
    import torch_npu

    # Blocked formats can report is_contiguous() but are not row-major.
    if any(torch_npu.get_npu_format(t) != 2 for t in (x, weight, indices)):
        return False
    # A layer slice of the four-dimensional Mamba pool retains plain NCHW
    # format (0).  Both NCHW and ND are unblocked; the strides above prove
    # their row-major indexing.  NZ and other blocked formats remain out.
    if torch_npu.get_npu_format(states) not in (0, 2):
        return False
    return True


def patch_qwen36_causal_conv_decode_fused():
    if os.getenv("SGLANG_FL_FUSED_CONV_DECODE", "1") != "1":
        return False
    if os.getenv("USE_FLAGGEMS", "1") != "1":
        return False
    if os.getenv("SGLANG_FL_FLAGOS_WHITELIST", "").strip():
        return False
    explicit_blacklist = os.getenv("SGLANG_FL_FLAGOS_BLACKLIST", "").strip()
    excluded = {item.strip() for item in explicit_blacklist.split(",")}
    if explicit_blacklist and "sum_dim" not in excluded:
        return False
    if "silu" in excluded:
        return False
    try:
        from sglang.srt.hardware_backend.npu.attention import (
            ascend_gdn_backend as module,
        )

        original = module.AscendGDNAttnBackend.forward_decode
        if getattr(original, _MARKER, False):
            return False
        if tuple(inspect.signature(original).parameters) != (
            "self",
            "layer",
            "forward_batch",
            "mixed_qkv",
            "a",
            "b",
            "kwargs",
        ):
            return False
        backend_source = inspect.getsource(original)
        update_source = inspect.getsource(module.causal_conv1d_update)
        replay_source = inspect.getsource(
            module.AscendMambaAttnBackendBase._replay_metadata
        )
        if not (
            "conv_states.transpose(1, 2).clone()" in backend_source
            and "conv_states[:] = conv_states_tmp.transpose(1, 2)" in backend_source
            and "torch_causal_conv1d_update_npu" in update_source
            and "conv_state_update = conv_state[conv_state_indices]" in update_source
            and "req_pool_indices[bs - num_padding :] = 0" in replay_source
            and "mamba_indices[bs - num_padding :] = 0" in replay_source
        ):
            return False
    except (ImportError, AttributeError, OSError, TypeError, ValueError) as exc:
        logger.warning("Fused decode convolution skipped: %s", exc)
        return False

    @functools.wraps(original)
    def forward_decode(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        states = layer_cache.conv[0]
        indices = self.forward_metadata.mamba_cache_indices
        if not _supported(mixed_qkv, states, layer, forward_batch, indices):
            return original(self, layer, forward_batch, mixed_qkv, a, b, **kwargs)
        coalesced_weight = os.getenv("SGLANG_FL_CONV_COALESCED_WEIGHT", "0") == "1"
        weight = layer.conv_weights if coalesced_weight else _transposed_weight(layer)
        if not getattr(self, "_sglang_fl_fused_conv_logged", False):
            logger.info(
                "Fused BF16 decode convolution active: input=%s pool=%s",
                tuple(mixed_qkv.shape),
                tuple(states.shape),
            )
            self._sglang_fl_fused_conv_logged = True
        mixed_qkv = fused_decode_conv(
            mixed_qkv, states, weight, indices, coalesced_weight=coalesced_weight
        )
        query, key, value = torch.split(
            mixed_qkv, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
        )
        bs = forward_batch.batch_size
        query = query.view(1, bs, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, bs, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, bs, layer.num_v_heads, layer.head_v_dim)
        ssm_states = layer_cache.temporal
        result = self.kernel_dispatcher.decode(
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=indices,
            query_start_loc=self.forward_metadata.query_start_loc,
        )
        self._track_mamba_state_decode(forward_batch, states, ssm_states, indices)
        return result

    setattr(forward_decode, _MARKER, True)
    module.AscendGDNAttnBackend.forward_decode = forward_decode
    logger.info(
        "Installed BF16 decode convolution fusion with explicit intermediate rounding"
    )
    return True
