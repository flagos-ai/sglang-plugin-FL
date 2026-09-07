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

"""Torch-native replacements for genuinely-called ``sgl_kernel_npu.norm`` symbols.

The zero-sgl_kernel_npu shim's ``_Dummy.__iter__`` yields nothing, so any model
that *actually calls* a shimmed ``split_qkv_*`` kernel dies on first unpack
(``ValueError: not enough values to unpack``). These implementations reproduce
the model's native ``forward_prepare`` sequence -- split, per-head RMSNorm,
RoPE -- so every model file that binds the symbol under ``if _is_npu:`` picks up
the real function with no per-file edits (the patch runs in ``load_plugin``,
before the model modules import).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_patched = False


def _per_head_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    head_dim: int,
    *,
    gemma_style: bool = False,
    cast_norm_to_bf16: bool = False,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """RMSNorm over the last ``head_dim`` of a flat ``(N, num_heads * head_dim)`` tensor.

    Mirrors the model-side ``apply_qk_norm`` path (per-head reshape, float32
    variance, rsqrt, weight, cast back). ``gemma_style`` follows GemmaRMSNorm,
    whose weight parameter stores 0 so the effective weight is ``1 + weight``
    (qwen3_next / qwen3_5 pass that zeros-param). minimax_m3 instead pre-adds
    the ones into its ``gemma_weight`` buffer and passes it as ``weight`` with
    ``cast_norm_to_bf16=True``.
    """
    orig_dtype = x.dtype
    xf = x.reshape(-1, head_dim).to(torch.float32)
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + eps)
    if cast_norm_to_bf16:
        xf = xf.to(torch.bfloat16)
        xf = xf * weight.to(torch.bfloat16)
    else:
        if gemma_style:
            xf = xf * (1.0 + weight.float())
        else:
            xf = xf * weight.float()
    if bias is not None:
        xf = xf + bias.to(xf.dtype)
    return xf.to(orig_dtype).reshape(x.shape)


def _apply_rope(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
    num_heads: int,
    head_dim: int,
    is_neox_style: bool = True,
) -> torch.Tensor:
    """Rotary-embed a tensor and return it in 3D head layout ``(N, num_heads, head_dim)``.

    Accepts a flat ``(N, num_heads * head_dim)`` input (forward_native layout);
    the 3D output is what ``torch_npu._npu_reshape_and_cache`` requires for the
    fresh k/v tensors (2D flat k/v makes the ATB op fail -- 507035 / setup
    failed), and q is reshaped by the backend anyway, so 3D is safe for all three.

    ``cos`` / ``sin`` carry the ``rotary_dim // 2`` distinct per-position values,
    in any shape whose first dim is ``N`` and whose trailing dims broadcast to
    ``(1, freq_dim)`` -- either a ``position_cos`` slice ``(N, 1, 1, freq_dim)``
    or a plain chunked cache ``(N, freq_dim)``.
    """
    freq_dim = rotary_dim // 2
    num_tokens = q.shape[0]
    q = q.reshape(num_tokens, num_heads, head_dim)
    rot, pas = q[..., :rotary_dim], q[..., rotary_dim:]
    cos = cos.reshape(num_tokens, 1, freq_dim).to(q.dtype)
    sin = sin.reshape(num_tokens, 1, freq_dim).to(q.dtype)
    if is_neox_style:
        x1, x2 = rot[..., :freq_dim], rot[..., freq_dim:]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        rot = torch.cat((o1, o2), dim=-1)
    else:
        x1, x2 = rot[..., 0::2], rot[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        rot = torch.stack((o1, o2), dim=-1).flatten(-2)
    return torch.cat((rot, pas), dim=-1)


def split_qkv_rmsnorm_rope(
    qkv: torch.Tensor,
    position_sin: torch.Tensor,
    position_cos: torch.Tensor,
    q_size: int,
    kv_size: int,
    head_dim: int,
    *,
    eps: float | None = None,
    q_weight: torch.Tensor | None = None,
    k_weight: torch.Tensor | None = None,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused split + qk-norm + RoPE (qwen3 / qwen3_moe / dflash / llama / glm4_moe).

    Reproduces the native ``split -> apply_qk_norm -> rotary_emb`` sequence.
    Callers without a qk norm (llama, glm4_moe with use_qk_norm=False) pass
    eps/weights as None and the norm is skipped. ``position_cos``/``position_sin``
    are the BSNH caches from ``get_cos_sin_with_position`` ``(N, 1, 1, rotary_dim)``
    whose leading halves hold the distinct per-position cos/sin values.
    """
    rotary_dim = position_cos.shape[-1]
    num_q_heads = q_size // head_dim
    num_kv_heads = kv_size // head_dim
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    # 3D k/v head layout: reshape_and_cache requires it (see _apply_rope note).
    v = v.reshape(-1, num_kv_heads, head_dim)
    if eps is not None and q_weight is not None:
        q = _per_head_rmsnorm(q, q_weight, eps, head_dim, bias=q_bias)
        k = _per_head_rmsnorm(k, k_weight, eps, head_dim, bias=k_bias)
    cos = position_cos[..., : rotary_dim // 2]
    sin = position_sin[..., : rotary_dim // 2]
    q = _apply_rope(q, cos, sin, rotary_dim, num_q_heads, head_dim, is_neox_style)
    k = _apply_rope(k, cos, sin, rotary_dim, num_kv_heads, head_dim, is_neox_style)
    return q, k, v


def split_qkvgate_gemma_rmsnorm_rope(
    qkv: torch.Tensor,
    position_sin: torch.Tensor,
    position_cos: torch.Tensor,
    q_size: int,
    kv_size: int,
    head_dim: int,
    rotary_dim: int | None = None,
    *,
    eps: float | None = None,
    q_weight: torch.Tensor | None = None,
    k_weight: torch.Tensor | None = None,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gated-attention split + gemma qk-norm + RoPE (qwen3_next / qwen3_5).

    The fused qkv projection carries twice the q width (q + gate); the gate is
    split off per head (native ``view(num_heads, -1) -> chunk``), and the q/k
    norms follow GemmaRMSNorm (weight stores 0, effective ``1 + weight``).
    ``rotary_dim`` is the explicit ``int(head_dim * partial_rotary_factor)`` the
    callers pass; it equals ``position_cos.shape[-1]``.
    """
    if rotary_dim is None:
        rotary_dim = position_cos.shape[-1]
    num_q_heads = q_size // head_dim
    num_kv_heads = kv_size // head_dim
    q_gate, k, v = qkv.split([q_size * 2, kv_size, kv_size], dim=-1)
    q, gate = torch.chunk(q_gate.reshape(-1, num_q_heads, 2 * head_dim), 2, dim=-1)
    q = q.reshape(-1, q_size)
    gate = gate.reshape(-1, q_size)
    # gate stays FLAT: qwen3_next consumes it as ``attn_output * sigmoid(gate)``
    # with flat attn_output, so 3D would break the broadcast. v needs 3D for
    # reshape_and_cache (see _apply_rope note).
    v = v.reshape(-1, num_kv_heads, head_dim)
    if eps is not None and q_weight is not None:
        q = _per_head_rmsnorm(q, q_weight, eps, head_dim, gemma_style=True, bias=q_bias)
        k = _per_head_rmsnorm(k, k_weight, eps, head_dim, gemma_style=True, bias=k_bias)
    cos = position_cos[..., : rotary_dim // 2]
    sin = position_sin[..., : rotary_dim // 2]
    q = _apply_rope(q, cos, sin, rotary_dim, num_q_heads, head_dim, is_neox_style)
    k = _apply_rope(k, cos, sin, rotary_dim, num_kv_heads, head_dim, is_neox_style)
    return q, k, v, gate


def split_qkv_rmsnorm_rope_pos_cache_half_npu(
    qkv: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
    q_size: int | None = None,
    kv_size: int | None = None,
    head_dim: int | None = None,
    *,
    eps: float | None = None,
    q_weight: torch.Tensor | None = None,
    k_weight: torch.Tensor | None = None,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    rope_dim: int | None = None,
    cast_norm_to_bf16: bool = False,
    is_neox_style: bool = True,
    # minimax_m3 names the qkv/size args differently (keyword-only call).
    input_tensor: torch.Tensor | None = None,
    q_hidden_size: int | None = None,
    kv_hidden_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split + qk-norm + RoPE from ``cos_sin_cache`` + ``positions`` (llada2 / minimax_m3).

    The cache is indexed directly instead of via the precomputed BSNH
    ``position_cos/sin``; the first ``rope_dim`` dims of each head are rotated
    (partial "half" rotary). minimax_m3 passes its pre-added ``gemma_weight``
    buffer as ``q_weight``/``k_weight`` with ``cast_norm_to_bf16=True``.
    """
    if input_tensor is not None:
        qkv = input_tensor
    if q_hidden_size is not None:
        q_size = q_hidden_size
    if kv_hidden_size is not None:
        kv_size = kv_hidden_size
    if rope_dim is None:
        rope_dim = cos_sin_cache.shape[-1]
    num_q_heads = q_size // head_dim
    num_kv_heads = kv_size // head_dim
    cos_sin = cos_sin_cache.index_select(0, positions.flatten())
    cos = cos_sin[..., : rope_dim // 2]
    sin = cos_sin[..., rope_dim // 2 :]
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    v = v.reshape(-1, num_kv_heads, head_dim)
    if eps is not None and q_weight is not None:
        q = _per_head_rmsnorm(
            q, q_weight, eps, head_dim, cast_norm_to_bf16=cast_norm_to_bf16, bias=q_bias
        )
        k = _per_head_rmsnorm(
            k, k_weight, eps, head_dim, cast_norm_to_bf16=cast_norm_to_bf16, bias=k_bias
        )
    q = _apply_rope(q, cos, sin, rope_dim, num_q_heads, head_dim, is_neox_style)
    k = _apply_rope(k, cos, sin, rope_dim, num_kv_heads, head_dim, is_neox_style)
    return q, k, v


def split_qkv_tp_rmsnorm_rope(
    *,
    input: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    tp_world: int | None = None,
    tp_group: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split + qk-norm + RoPE with TP-sharded qk-norm weights (minimax_m2).

    The model already chunks ``cos_sin_cache`` into plain ``(N, freq_dim)``
    cos/sin. ``MiniMaxM2RMSNormTP`` computes a purely local per-head variance
    when ``tp_world == 1``; for ``tp_world > 1`` it would all-reduce the
    variance across the attn TP group, which this torch-native version does not
    reproduce (single-device runs are exact).
    """
    num_q_heads = q_hidden_size // head_dim
    num_kv_heads = kv_hidden_size // head_dim
    q, k, v = input.split([q_hidden_size, kv_hidden_size, kv_hidden_size], dim=-1)
    v = v.reshape(-1, num_kv_heads, head_dim)
    if eps is not None and q_weight is not None:
        q = _per_head_rmsnorm(q, q_weight, eps, head_dim)
        k = _per_head_rmsnorm(k, k_weight, eps, head_dim)
    q = _apply_rope(q, cos, sin, rotary_dim, num_q_heads, head_dim, True)
    k = _apply_rope(k, cos, sin, rotary_dim, num_kv_heads, head_dim, True)
    return q, k, v


class _AllocExtendKernel:
    """Torch-native stand-in for the shimmed ``alloc_extend_kernel``.

    ``NPUPagedTokenToKVPoolAllocator.alloc_extend`` launches it with Triton grid
    syntax (``alloc_extend_kernel[(bs,)](...)``). On a shim ``_Dummy`` that
    launch silently no-ops and ``out_indices`` (a fresh ``torch.empty``) stays
    as uninitialized memory; the values later become the KV slot indices for
    ``_npu_reshape_and_cache``, which then writes to OOB DDR and trips the MTE
    exception 507035. The ``__getitem__``-returns-self trampoline keeps the
    grid-subscript call site intact and lands the arguments on
    ``alloc_extend_naive`` -- the exact code the >=200-new-pages branch already
    runs, which is correct. The ``bs_upper`` constexpr (arg 6) and
    ``max_num_extend_tokens`` (arg 8) are bounds hints the naive version does
    not need.
    """

    def __init__(self, alloc_extend_naive):
        self._naive = alloc_extend_naive

    def __getitem__(self, grid):
        return self

    def __call__(self, *args):
        (
            prefix_lens,
            seq_lens,
            last_loc,
            free_pages,
            out_indices,
            _bs_upper,
            page_size,
            *_,
        ) = args
        return self._naive(
            prefix_lens,
            seq_lens,
            last_loc,
            free_pages,
            out_indices,
            page_size,
            prefix_lens.device,
        )


def patch_npu_kernel_stubs() -> None:
    """Bind the real implementations onto the zero-shim ``sgl_kernel_npu`` stubs.

    Model files bind the symbols at module import time (``if _is_npu:
    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import ...``); the patch
    runs in ``load_plugin`` step 5, before any model module imports, so setting
    the real functions as module attributes makes every call site pick them up.
    The stub module files exist (each just ``from sgl_kernel_npu import
    __getattr__``), so a plain ``import`` reaches them and the attribute set
    shadows the module ``__getattr__``.
    """
    global _patched
    if _patched:
        return
    _patched = True

    try:
        import sgl_kernel_npu.norm.split_qkv_rmsnorm_rope as _sqr
        import sgl_kernel_npu.norm.split_qkv_rmsnorm_rope_pos_cache_half_npu as _sqp
        import sgl_kernel_npu.norm.split_qkv_tp_rmsnorm_rope as _sqtp
        import sgl_kernel_npu.mem_cache.allocator as _alloc
        from sglang.srt.mem_cache.allocator import alloc_extend_naive
    except Exception as e:  # pragma: no cover - guard only
        logger.warning("Ascend npu_kernel_stubs patch skipped: %s", e)
        return

    _sqr.split_qkv_rmsnorm_rope = split_qkv_rmsnorm_rope
    _sqr.split_qkvgate_gemma_rmsnorm_rope = split_qkvgate_gemma_rmsnorm_rope
    _sqp.split_qkv_rmsnorm_rope_pos_cache_half_npu = (
        split_qkv_rmsnorm_rope_pos_cache_half_npu
    )
    _sqtp.split_qkv_tp_rmsnorm_rope = split_qkv_tp_rmsnorm_rope
    _alloc.alloc_extend_kernel = _AllocExtendKernel(alloc_extend_naive)

    logger.info("Ascend npu_kernel_stubs patch applied")
