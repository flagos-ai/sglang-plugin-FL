"""Kunlunxin: route GDN causal_conv1d (prefill + decode) to the klx sgl_kernel.

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
belt-and-suspenders step (see ``patch_causal_conv1d_native``).

Two independent problems make the stock path unusable on Kunlun, and both the
prefill dispatcher-fallback and the decode triton binding hit them:

1. Kunlun's Triton XPU backend cannot compile ``_causal_conv1d_fwd_kernel`` /
   ``_causal_conv1d_update_kernel``:
       fp16: AssertionError('mismatched type for col0 ... bf16 vs fp16')
       bf16: 'arith.muli' op requires the same type ...
             -> OutOfResources: uni_sram PassManager::run failed
2. The stock cuda ``sgl_kernel.causal_conv1d_*`` binding needs extra trailing
   args vs sglang 0.5.12's call convention:
       TypeError: causal_conv1d_fwd() missing 2 required positional arguments ...

So we replace both functions with kunlunxin-native wrappers:

  * PREFILL -> ``fla_xpu_ops.causal_conv1d_fwd`` (the xfla xpu op). Runs NATIVELY
    in the model dtype (bf16/fp16), NWC layout, ``is_ncw=False``. sglang passes
    ``x`` as ``(dim, tokens)`` NCW packed and ``conv_states`` as
    ``(num_cache_lines, dim, state_len)`` NCW; we transpose both to NWC and back.
    (No fp32 shadow — the xfla kernel handles bf16/fp16 directly, unlike the
    klx ``sgl_kernel.causal_conv1d_fwd`` which is fp32-only.)
  * DECODE -> ``sgl_kernel.causal_conv1d_update`` (klx), fp32 shadow + NWC
    (``is_ncw=False``); see ``_kunlun_causal_conv1d_update``. (The xfla package's
    ``causal_conv1d_update`` is a pure-python ref that drops conv_state_indices,
    so decode stays on the klx kernel.)
"""

import itertools
import logging
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)
_applied = False


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
    """GDN prefill causal_conv1d_fn -> xfla ``fla_xpu_ops.causal_conv1d_fwd``.

    Unlike the klx ``sgl_kernel.causal_conv1d_fwd`` (fp32-only, is_ncw=True), the
    xfla op runs NATIVELY in the model dtype (bf16/fp16) — no fp32 shadow — and
    expects the NWC layout. sglang's ``gdn_backend`` passes ``x`` as
    ``(dim, total_tokens)`` NCW packed (``mixed_qkv.transpose(0, 1)``) and
    ``conv_states`` as ``(num_cache_lines, dim, state_len)`` NCW; we transpose
    both into NWC for the kernel, then transpose the output / updated state back.
    """
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import PAD_SLOT_ID

    # Deferred, kunlunxin-local side-effect import (registers the xfla xpu ops).
    import fla_xpu_ops

    if pad_slot_id is None:
        pad_slot_id = PAD_SLOT_ID
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    silu_activation = activation in ["silu", "swish"]

    # x: (dim, total_tokens) NCW  ->  x_flat: (total_tokens, dim) NWC
    x_flat = x.transpose(0, 1).contiguous()
    run_dtype = x_flat.dtype

    # The kernel casts weight to x.dtype internally; do it up front so bias/weight
    # match. No fp32 shadow — bf16/fp16 run natively on the xfla path.
    if weight.dtype != run_dtype:
        weight = weight.to(run_dtype)
    bias = bias.contiguous() if bias is not None else None

    # cu_seqlens: device + cpu int32 (the kernel slices each request out of the
    # packed batch using both).
    if query_start_loc is not None:
        qsl = query_start_loc.to(torch.int32)
        qsl_cpu = query_start_loc.detach().to(dtype=torch.int32, device="cpu")
    else:
        if seq_lens_cpu is None:
            seq_lens_cpu = [x_flat.shape[0]]
        elif not isinstance(seq_lens_cpu, list):
            seq_lens_cpu = seq_lens_cpu.tolist()
        prefix = [0] + list(itertools.accumulate(seq_lens_cpu))
        qsl_cpu = torch.tensor(prefix, dtype=torch.int32)
        qsl = qsl_cpu.to(x.device)

    num_seqs = qsl.numel() - 1

    if cache_indices is not None:
        cache_indices = cache_indices.to(torch.int32)
    else:
        cache_indices = torch.arange(num_seqs, device=x.device, dtype=torch.int32)

    # gdn_backend passes has_initial_state as a bool tensor (extend_prefix_lens>0);
    # the kernel wants an int32 per-seq flag.
    if has_initial_state is not None:
        has_initial_state = has_initial_state.to(torch.int32)
    else:
        has_initial_state = torch.zeros(num_seqs, device=x.device, dtype=torch.int32)

    # conv_states: (num_cache_lines, dim, state_len) NCW -> NWC (…, state_len, dim)
    cs_nwc = (
        conv_states.transpose(1, 2).contiguous() if conv_states is not None else None
    )

    out_flat = torch.empty_like(x_flat)

    fla_xpu_ops.causal_conv1d_fwd(
        x_flat,
        weight,
        qsl,
        qsl_cpu,
        cache_indices,
        has_initial_state,
        silu_activation,
        False,  # is_ncw = False (NWC layout)
        pad_slot_id,
        bias,
        cs_nwc,
        1,  # dilation
        out_flat,
    )

    # Write the updated NWC conv state back into the stock NCW cache in place.
    if conv_states is not None:
        conv_states.copy_(cs_nwc.transpose(1, 2))

    # out_flat: (total_tokens, dim) NWC -> (dim, total_tokens) NCW for the caller
    # (gdn_backend does .transpose(0, 1)[:seq_len] on the return value).
    return out_flat.transpose(0, 1)


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
    """GDN decode causal_conv1d_update -> klx-native sgl_kernel.

    Mirrors the prefill patch's motivation: the stock cuda-path
    ``causal_conv1d_update`` binds sglang's Triton kernel, which the Kunlun XPU
    Triton backend cannot compile (uni_sram OutOfResources). Route to the klx
    ``sgl_kernel.causal_conv1d_update`` instead.

    The klx kernel needs the extra trailing ``is_ncw`` / ``pad_slot_id`` args and
    a seqlen axis on ``x``. Decode is single-token: SGLang passes ``x`` as 2D
    ``(batch, dim)``; add the NCW seqlen axis (``-> (batch, dim, 1)``) to match
    the ``is_ncw=True`` layout the prefill path uses on the shared conv_states
    buffer, then squeeze it back. The kernel updates ``x`` and ``conv_state`` in
    place and we return ``x`` (SGLang consumes the return value).
    """
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import PAD_SLOT_ID

    # Deferred, kunlunxin-local side-effect import (registers torch.ops.sgl_kernel.*).
    import sgl_kernel  # noqa: F401

    if pad_slot_id is None:
        pad_slot_id = PAD_SLOT_ID
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    # Layout + dtype bridging between stock SGLang's conv cache and the klx
    # kernel. Verified empirically (tests/test_kunlunxin_kernel.py): the klx
    # causal_conv1d_update kernel accepts ONLY:
    #   * dtype float32 — fp16/bf16 raise "expected scalar type Float but
    #     found Half/BFloat16"; and
    #   * is_ncw=False (NWC layout) — x as (batch, seqlen, dim), conv_state as
    #     (num_cache_lines, state_len, dim). is_ncw=True hits an internal
    #     transpose that faults even in fp32.
    # Stock SGLang allocates the cache as (num_cache_lines, dim, state_len) in
    # the model's bf16 dtype and passes x as 2D (batch, dim) fp16. So we build a
    # transposed fp32 shadow of the cache, run the kernel, and write the updated
    # state back. The reference xSGL fork instead allocates the cache transposed
    # via XSGL_TRANSPOSE_CONV_STATE, which we cannot do without touching the
    # common memory-pool code.
    run_dtype = torch.float32
    out_dtype = x.dtype

    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(1)  # (batch, dim) -> (batch, seqlen=1, dim) NWC

    x = x.to(run_dtype).contiguous()
    weight = weight.to(run_dtype)
    if bias is not None:
        bias = bias.to(run_dtype)

    # (num_cache_lines, dim, state_len) -> NWC (num_cache_lines, state_len, dim)
    cs = conv_state.transpose(1, 2).to(run_dtype).contiguous()

    klx_causal_conv1d_update = torch.ops.sgl_kernel.causal_conv1d_update.default
    klx_causal_conv1d_update(
        x,
        cs,
        weight,
        bias,
        activation in ["silu", "swish"],
        cache_seqlens,
        conv_state_indices,
        intermediate_conv_window,
        False,  # is_ncw: klx update kernel needs NWC layout
        pad_slot_id,
    )

    # Write the updated NWC state back into the stock (dim, state_len) cache.
    conv_state.copy_(cs.transpose(1, 2).to(conv_state.dtype))

    if unsqueeze:
        x = x.squeeze(1)
    if x.dtype != out_dtype:
        x = x.to(out_dtype)
    return x


def patch_causal_conv1d():
    """Patch causal_conv1d_triton.{causal_conv1d_fn,causal_conv1d_update}.

    Patching at the triton module seam (rather than gdn_backend directly) makes
    both consumers pick up our klx-signature wrappers:
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
