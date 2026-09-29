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

"""Kunlunxin: route GDN causal_conv1d (prefill + decode) to pure PyTorch.

Qwen3.5/3.6 is a hybrid model: its linear-attention (GDN) layers run a short
causal depthwise conv via ``causal_conv1d_fn`` (prefill) / ``causal_conv1d_update``
(decode) before the gated-delta-rule SSM.

We patch at the ``causal_conv1d_triton`` module seam, i.e. rebind
``sglang.srt.layers.attention.mamba.causal_conv1d_triton.causal_conv1d_fn`` and
``.causal_conv1d_update``. Consumers ``from``-import these names, so this patch
must run BEFORE they import (plugin-activation time, which it does) — then:
  * the cuda-compat dispatcher ``causal_conv1d.py`` picks up our functions as its
    ``_causal_conv1d_fn_triton`` / ``_causal_conv1d_update_triton`` fallbacks, and
  * ``gdn_backend`` (which on ``is_cuda()`` keeps ``causal_conv1d_update`` bound
    to the pure-triton import) picks up our update fn directly.
For already-imported consumers we also rebind their local names as a
belt-and-suspenders step.

Why not stock Triton / sgl_kernel / fla_xpu_ops:

1. Kunlun's Triton XPU backend cannot compile ``_causal_conv1d_fwd_kernel`` /
   ``_causal_conv1d_update_kernel`` (dtype / SRAM failures).
2. Stock cuda ``sgl_kernel.causal_conv1d_*`` call convention mismatches sglang
   0.5.12; klx fwd is fp32-only.
3. ``fla_xpu_ops`` is an optional vendor wheel; empty stacks may not ship it.

Both prefill and decode therefore use the same NCW windowed depthwise formula
(``unfold`` + elementwise ``mul`` + ``sum``, matching decode's seqlen=1 path).
We deliberately avoid ``F.conv1d``: on Kunlun with FlagGems enabled it is
rewritten through a broken Triton/conv path (logs show
``conv2d_forward_kernel`` / ``Removed N operations``) and corrupts GDN prefill.
"""

import itertools
import logging
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)
_applied = False


def _depthwise_causal_windows_ncw(
    x_ncw: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Depthwise causal conv on NCW ``(..., dim, length)`` without ``F.conv1d``.

    ``weight`` is ``(dim, width)``. Output length is ``length - width + 1``.
    Same math as decode's ``(window * w).sum(-1)`` for each time step.
    """
    width = weight.shape[1]
    # (..., dim, n_out, width)
    windows = x_ncw.unfold(dimension=-1, size=width, step=1)
    w = weight.to(dtype=x_ncw.dtype).view(
        *([1] * (windows.dim() - 3)), weight.shape[0], 1, width
    )
    y = (windows * w).sum(dim=-1)
    if bias is not None:
        b = bias.to(dtype=y.dtype)
        y = y + b.view(*([1] * (y.dim() - 2)), -1, 1)
    return y


def _silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU without relying on FlagGems ``silu`` / ``sigmoid`` kernels."""
    return x * torch.sigmoid(x)


def _kunlun_causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: Optional[int] = None,
    seq_lens_cpu: Optional[List[int]] = None,
):
    """GDN prefill causal_conv1d_fn — pure PyTorch (no fla_xpu_ops / sgl_kernel).

    ``x`` is ``(dim, total_tokens)`` NCW packed (gdn_backend does
    ``mixed_qkv.transpose(0, 1)``). Sequences are sliced by ``query_start_loc``
    / ``seq_lens_cpu``. Each sequence optionally seeds from
    ``conv_states[cache_indices[i]]`` when ``has_initial_state[i]`` is true,
    then depthwise-convolves with ``weight`` of shape ``(dim, width)`` and
    writes back the trailing ``width-1`` tokens into ``conv_states``.
    """
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import PAD_SLOT_ID

    if pad_slot_id is None:
        pad_slot_id = PAD_SLOT_ID
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    # x: (dim, total_tokens) NCW
    if x.dim() != 2:
        raise ValueError(
            f"expected packed x of shape (dim, tokens), got {tuple(x.shape)}"
        )
    dim, total_tokens = x.shape
    width = weight.shape[1]
    state_len = width - 1
    assert weight.shape == (dim, width)

    dtype_in = x.dtype
    w = weight.to(dtype=dtype_in)
    b = bias.to(dtype=dtype_in) if bias is not None else None

    if query_start_loc is not None:
        qsl_cpu = query_start_loc.detach().to(dtype=torch.int64, device="cpu").tolist()
    else:
        if seq_lens_cpu is None:
            seq_lens_list = [total_tokens]
        elif not isinstance(seq_lens_cpu, list):
            seq_lens_list = list(seq_lens_cpu)
        else:
            seq_lens_list = seq_lens_cpu
        qsl_cpu = [0] + list(itertools.accumulate(seq_lens_list))

    num_seqs = len(qsl_cpu) - 1
    if qsl_cpu[-1] != total_tokens:
        raise ValueError(
            f"query_start_loc ends at {qsl_cpu[-1]} but x has {total_tokens} tokens"
        )

    if cache_indices is not None:
        cache_idx_list = (
            cache_indices.detach().to(dtype=torch.int64, device="cpu").tolist()
        )
    else:
        cache_idx_list = list(range(num_seqs))

    if has_initial_state is not None:
        has_init_list = (
            has_initial_state.detach().to(dtype=torch.bool, device="cpu").tolist()
        )
    else:
        has_init_list = [False] * num_seqs

    out = torch.empty_like(x)

    for i in range(num_seqs):
        start, end = int(qsl_cpu[i]), int(qsl_cpu[i + 1])
        seqlen = end - start
        if seqlen <= 0:
            continue

        cache_i = int(cache_idx_list[i]) if i < len(cache_idx_list) else i
        if cache_i == pad_slot_id:
            out[:, start:end].zero_()
            continue

        x_seq = x[:, start:end]  # (dim, seqlen)

        if conv_states is not None and has_init_list[i]:
            prefix = conv_states[cache_i].to(dtype=dtype_in)  # (dim, state_len)
        else:
            prefix = torch.zeros((dim, state_len), dtype=dtype_in, device=x.device)

        # (dim, state_len + seqlen); n_out = seqlen when state_len == width - 1
        x_cat = torch.cat([prefix, x_seq], dim=-1)
        y = _depthwise_causal_windows_ncw(x_cat, w, b)
        if y.shape[-1] != seqlen:
            y = y[..., -seqlen:]

        if activation in ("silu", "swish"):
            y = _silu(y)

        out[:, start:end] = y.to(dtype=dtype_in)

        if conv_states is not None:
            # Trailing width-1 tokens of the effective input stream.
            new_state = (
                x_cat[:, -state_len:] if state_len > 0 else conv_states[cache_i]
            )
            conv_states[cache_i].copy_(new_state.to(dtype=conv_states.dtype))

    return out


def _kunlun_causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    pad_slot_id: Optional[int] = None,
):
    """Decode causal_conv1d_update — pure PyTorch, CUDA-graph safe.

    Avoids boolean indexing (``x[valid]``) which breaks Graph capture on Kunlun
    (FlagGems / XPU advanced-index path tried to allocate terabytes).
    """
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import PAD_SLOT_ID

    if pad_slot_id is None:
        pad_slot_id = PAD_SLOT_ID
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if cache_seqlens is not None:
        raise NotImplementedError(
            "cache_seqlens circular buffer not needed for GDN decode"
        )
    if intermediate_conv_window is not None:
        raise NotImplementedError(
            "intermediate_conv_window (speculative) not supported"
        )

    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)  # (B, D, 1) NCW
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert weight.shape == (dim, width)

    if conv_state_indices is not None:
        idx = conv_state_indices.to(dtype=torch.long)
        valid = idx != pad_slot_id  # [B], stay on device — no .any()/.item()
        safe_idx = torch.where(valid, idx, torch.zeros_like(idx))
        cs_gathered = conv_state[safe_idx]  # [B, D, state_len]
    else:
        valid = None
        safe_idx = None
        cs_gathered = conv_state

    # Process the full batch (fixed shape → Graph-friendly).
    x_new = torch.cat([cs_gathered, x], dim=-1)  # (B, D, state_len + seqlen)
    cs_updated = x_new[:, :, -state_len:]
    w = weight.to(dtype=x_new.dtype)

    # Same windowed formula as prefill / seqlen=1 decode — avoid F.conv1d
    # (FlagGems conv path is incorrect on Kunlun).
    out = _depthwise_causal_windows_ncw(
        x_new.to(w.dtype),
        w,
        bias.to(w.dtype) if bias is not None else None,
    )
    if out.shape[-1] != seqlen:
        out = out[..., -seqlen:]

    if activation in ("silu", "swish"):
        out = _silu(out)
    out = out.to(dtype=dtype_in)

    if conv_state_indices is not None:
        # Identity writeback for pad rows so clamped index 0 is not corrupted
        # when a pad slot and a real request both map to the same safe_idx.
        valid_b = valid.view(batch, 1, 1)
        cs_store = torch.where(
            valid_b, cs_updated.to(dtype=conv_state.dtype), cs_gathered
        )
        conv_state.index_copy_(0, safe_idx, cs_store)
        out = out * valid_b.to(dtype=out.dtype)
    else:
        conv_state.copy_(cs_updated.to(dtype=conv_state.dtype))

    if unsqueeze:
        out = out.squeeze(-1)
    return out


def patch_causal_conv1d():
    """Patch causal_conv1d_triton.{causal_conv1d_fn,causal_conv1d_update}.

    Patching at the triton module seam (rather than gdn_backend directly) makes
    both consumers pick up our PyTorch wrappers:
      * the cuda-compat dispatcher's ``_causal_conv1d_fn_triton`` /
        ``_causal_conv1d_update_triton`` fallbacks (prefill non-contig path), and
      * ``gdn_backend.causal_conv1d_update`` (decode; kept bound to the pure
        triton import on ``is_cuda()``).
    This runs at plugin-activation time, before those modules import the names.
    For any module that already imported them we rebind its local names too.
    """
    global _applied
    if _applied:
        return

    import sys

    from sglang.srt.layers.attention.mamba import causal_conv1d_triton

    causal_conv1d_triton.causal_conv1d_fn = _kunlun_causal_conv1d_fn
    causal_conv1d_triton.causal_conv1d_update = _kunlun_causal_conv1d_update

    # Belt-and-suspenders: rebind local names in any already-imported consumer.
    disp = sys.modules.get("sglang.srt.layers.attention.mamba.causal_conv1d")
    if disp is not None:
        disp._causal_conv1d_fn_triton = _kunlun_causal_conv1d_fn
        disp._causal_conv1d_update_triton = _kunlun_causal_conv1d_update

    gdn = sys.modules.get("sglang.srt.layers.attention.linear.gdn_backend")
    if gdn is not None:
        gdn.causal_conv1d_update = _kunlun_causal_conv1d_update

    _applied = True
