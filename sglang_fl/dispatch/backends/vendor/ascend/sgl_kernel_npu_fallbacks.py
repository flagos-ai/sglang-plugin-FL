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

"""Operators SGLang imports from ``sgl_kernel_npu`` on the operations Runtime.

The operations Runtime (``flagos-runtime-ascend-cann9.2.1-910c``) ships no
``sgl_kernel_npu`` wheel; SGLang 0.5.18 imports it module-level while loading
the NPU attention backends and the MoE top-k. The import guard
(``vendor/ascend/import_guard.py``) lets those imports succeed with a
placeholder; the names below would still fail loudly when called, so this
table maps placeholder module paths to real implementations taken from what
the platform already ships:

* plain PyTorch for the small math kernels, and
* the Triton kernels SGLang itself ships under ``sglang.kernels.ops`` -- the
  same code its non-vendor NPU paths run, so the GDN state layout stays the
  Triton kernel's K,V layout everywhere.

The first batch covers the call sites the CI model matrix reaches
(``tests/platforms/ascend.yaml``: qwen3 dense and qwen3_6 GDN+MoE, eager):

* ``norm.l1_norm.l1_norm`` -- MoE top-k renormalization
  (``hardware_backend/npu/moe/topk.py``);
* ``fla.fused_gdn_gating.fused_gdn_gating_npu`` -- GDN gating
  (``ascend_gdn_backend.py``, ``linear/gdn_backend.py``);
* ``fla.chunk.chunk_gated_delta_rule_npu`` -- GDN chunked prefill, also the
  name ``gdn_triton.py`` binds for its NPU chunk kernel;
* ``fla.fused_sigmoid_gating_recurrent.fused_sigmoid_gating_delta_rule_update_npu``
  -- GDN fused decode, bound the same way by ``gdn_triton.py``;
* ``mamba.mamba_state_update_triton.move_intermediate_cache`` -- the MTP
  state move the real-NPU probe ``examples/ascend_compat_probe.py``
  exercises; the PyTorch kernel stand-in keeps the plugin's CANN 8.5
  multibuffer patch (``patches/mamba_state_update.py``) applicable;
* ``torch.ops.npu.causal_conv1d`` -- the depthwise causal convolution the GDN
  backend launches on every step (registered by the vendor wheel's native
  extension; re-registered here as a torch library op backed by plain PyTorch
  when no wheel registered it).

Deliberately NOT implemented (the placeholder fails loudly if reached, the
same policy as Hygon's table): the speculative paths the Ascend CI matrix
does not exercise -- ``fused_gdn_gating_kernel_without_sigmoid``,
``causal_conv1d_verify``, ``move_intermediate_cache_kda``,
``conv_state_rollback``, the ``kda_*`` family and ``solve_tril`` -- and the
optional fast paths SGLang selects through
``try/except ImportError`` (``fused_rope_qk_mqa``), which the guard keeps
absent so SGLang's generic path takes over.

A really installed ``sgl_kernel_npu`` always wins: the import guard stays out
of its way, the torch op registration is skipped, and this table is never
consulted.
"""

from __future__ import annotations

from typing import Optional

import torch


def l1_norm(input: torch.Tensor) -> torch.Tensor:
    """Normalize rows by their L1 norm, like ``sgl_kernel_npu.norm.l1_norm``.

    Reached from ``hardware_backend/npu/moe/topk.py`` for ``renormalize``
    routing, where rows are non-negative scores; the fp32 intermediate keeps
    the division exact for bf16/fp16 inputs. A fully zero row has no direction
    to normalize: the denominator swap keeps it finite (zeros, the 0/0 the
    vendor kernel emits), and MoE routing masks such rows downstream.
    """
    computed = input.float()
    denominator = computed.abs().sum(dim=-1, keepdim=True)
    safe = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    return (computed / safe).to(input.dtype)


def fused_gdn_gating_npu(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
):
    """GDN gating: ``g = -exp(A_log) * softplus(a + dt_bias)``, ``beta = sigmoid(b)``.

    Same contract as SGLang's Triton ``fused_gdn_gating`` (the vendor kernel's
    NPU name): fp32 compute, outputs shaped ``(1, tokens, heads)``. Delegates
    to SGLang's own kernel so decode (``a`` is ``(batch, heads)``) and extend
    (``a`` is ``(tokens, heads)``) both work.
    """
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating

    return fused_gdn_gating(A_log, a, b, dt_bias, beta=beta, threshold=threshold)


def chunk_gated_delta_rule_npu(*args, **kwargs):
    """GDN chunked prefill; the name ``gdn_triton.py`` binds on NPU.

    Same signature and return contract as SGLang's Triton
    ``chunk_gated_delta_rule``. One bridge: the NPU call form
    (``gdn_triton.py``) gathers the per-request states
    (``ssm_states[cache_indices]``) and passes no ``initial_state_indices``,
    while the Triton kernel loads its slot ids unconditionally
    (``tl.load(initial_state_indices + i_n)``), so numbering the gathered
    tensor makes slot ``i`` read request ``i``'s state.
    """
    initial_state = kwargs.get("initial_state")
    if (
        not args
        and initial_state is not None
        and kwargs.get("initial_state_indices") is None
    ):
        kwargs["initial_state_indices"] = torch.arange(
            initial_state.shape[0], device=initial_state.device
        )
    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule

    return chunk_gated_delta_rule(*args, **kwargs)


def fused_sigmoid_gating_delta_rule_update_npu(*args, **kwargs):
    """GDN fused decode update; bound by ``gdn_triton.py`` on NPU.

    Same signature and return contract as SGLang's Triton
    ``fused_sigmoid_gating_delta_rule_update``.
    """
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    return fused_sigmoid_gating_delta_rule_update(*args, **kwargs)


def _next_power_of_2(value: int) -> int:
    raised = 1
    while raised < value:
        raised *= 2
    return raised


class _MoveCacheDynamicLastKernel:
    """PyTorch stand-in for the Triton kernel ``move_cache_dynamic_last_kernel_h_block``.

    The vendor object is a ``triton.jit`` kernel launched as
    ``kernel[grid](**kwargs)``; the plugin's CANN 8.5 patch
    (``patches/mamba_state_update.py``) wraps its ``.run`` to force
    ``multibuffer=False`` for the MTP tile that overflows the 910C unified
    buffer. This stand-in keeps that launch surface -- ``__getitem__`` routes
    through ``run``, so the patch keeps applying -- and computes the same
    assignment in plain PyTorch. ``run`` accepts and ignores the vendor
    kernel's scheduling kwargs (strides, block sizes, ``multibuffer``): the
    vendor wrapper derives the strides from these same tensors, and the
    advanced indexing below follows the tensors' own layout, so the selected
    elements are identical.
    """

    def run(
        self,
        *,
        grid=None,
        dst_cache_ptr: torch.Tensor,
        src_cache_ptr: torch.Tensor,
        dst_indices_ptr: torch.Tensor,
        src_indices_ptr: torch.Tensor,
        last_steps_ptr: torch.Tensor,
        **_scheduling: int,
    ) -> None:
        # A negative last step is the skip marker: the kernel returns from
        # that program without touching the destination slot.
        valid = last_steps_ptr >= 0
        if bool(valid.any()):
            # Per valid i: dst[:, dst_indices[i]] = src[:, src_indices[i], last_steps[i]].
            # Duplicate destination indices keep the vendor kernel's own
            # last-writer-wins race; the CI call sites use unique indices.
            dst_cache_ptr[:, dst_indices_ptr[valid].to(torch.int64)] = src_cache_ptr[
                :,
                src_indices_ptr[valid].to(torch.int64),
                last_steps_ptr[valid].to(torch.int64),
            ]

    def __getitem__(self, grid):
        return lambda **kwargs: self.run(grid=grid, **kwargs)


move_cache_dynamic_last_kernel_h_block = _MoveCacheDynamicLastKernel()


def move_intermediate_cache(
    ssm_states: torch.Tensor,
    intermediate_state_cache: torch.Tensor,
    dst_indices_tensor: torch.Tensor,
    src_indices_tensor: torch.Tensor,
    last_steps_tensor: torch.Tensor,
    h_block_size: int = 2,
) -> torch.Tensor:
    """Move speculative intermediate states into the SSM state pool.

    Same contract as ``sgl_kernel_npu.mamba.mamba_state_update_triton``:
    ``ssm_states`` is ``(L, pool_size, H, V, K)`` and is updated in place,
    ``intermediate_state_cache`` is ``(L, requests, steps, H, V, K)``, and for
    every index ``i`` with ``last_steps[i] >= 0`` the copy is
    ``ssm_states[:, dst_indices[i]] =
    intermediate_state_cache[:, src_indices[i], last_steps[i]]``.
    ``h_block_size`` is the vendor kernel's tiling knob; the assignment it
    schedules does not depend on it. Returns ``ssm_states``.
    """
    if len(dst_indices_tensor) != len(last_steps_tensor):
        raise RuntimeError("Destination indices lengths must match")
    if len(src_indices_tensor) != len(last_steps_tensor):
        raise RuntimeError("Source indices lengths must match")

    _, _, _, _, dim_v, dim_k = intermediate_state_cache.shape
    move_cache_dynamic_last_kernel_h_block[(len(dst_indices_tensor),)](
        dst_cache_ptr=ssm_states,
        src_cache_ptr=intermediate_state_cache,
        dst_indices_ptr=dst_indices_tensor,
        src_indices_ptr=src_indices_tensor,
        last_steps_ptr=last_steps_tensor,
        H_BLOCK_SIZE=h_block_size,
        BLOCK_V=_next_power_of_2(dim_v),
        BLOCK_K=_next_power_of_2(dim_k),
    )
    return ssm_states


def _maybe_silu(output: torch.Tensor, activation_mode: int) -> torch.Tensor:
    if activation_mode == 1:
        return torch.nn.functional.silu(output)
    if activation_mode != 0:
        raise RuntimeError(
            f"npu::causal_conv1d fallback: unknown activation_mode {activation_mode}"
        )
    return output


def _causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: Optional[torch.Tensor],
    bias: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    activation_mode: int = 0,
    pad_slot_id: int = -1,
    run_mode: int = 0,
) -> torch.Tensor:
    """Depthwise causal conv1d with in-place conv-state advance, in plain PyTorch.

    Layouts follow the SGLang NPU call sites (``ascend_gdn_backend.py``):
    ``x`` is ``(total_tokens, dim)`` varlen-packed, ``weight`` is
    ``(width, dim)`` (the caller passes ``conv_weights.transpose(0, 1)``),
    ``conv_states`` is ``(batch, dim, width - 1)`` and is advanced in place
    (shift-left by the consumed tokens, then the newest window appended), and
    ``query_start_loc``/``cache_indices`` address the varlen batch. Rows whose
    ``cache_indices`` equals ``pad_slot_id`` are passed through without
    touching the state pool. ``activation_mode == 1`` appends SiLU, and
    accumulation is fp32 like the Triton reference kernels.

    ``run_mode`` only selects the vendor's decode/prefill fast paths; the math
    is the same per-sequence convolution, so one implementation serves both.
    """
    if num_accepted_tokens is not None:
        raise RuntimeError(
            "npu::causal_conv1d fallback has no num_accepted_tokens form"
        )
    if x.dim() != 2:
        raise RuntimeError(
            f"npu::causal_conv1d fallback expects a 2-D x, found {tuple(x.shape)}"
        )
    total_tokens, dim = x.shape
    width = weight.shape[0]
    if weight.shape[1] != dim:
        raise RuntimeError(
            f"npu::causal_conv1d fallback weight must be (width, dim)=({width}, {dim}), "
            f"found {tuple(weight.shape)}"
        )

    device = x.device
    if query_start_loc is None:
        query_start_loc = torch.tensor((0, total_tokens), device=device)
    batch = int(query_start_loc.numel()) - 1
    if cache_indices is None:
        cache_indices = torch.arange(batch, device=device)

    compute_f = torch.empty(total_tokens, dim, device=device, dtype=torch.float32)
    weight_f = weight.float()
    bias_f = bias.float() if bias is not None else None
    lengths = (query_start_loc[1:] - query_start_loc[:-1]).tolist()

    single_token = all(length == 1 for length in lengths)
    if single_token:
        # Decode: every sequence consumes one token, so the whole batch
        # advances in one vectorized step. The existing pool state is always
        # the initial state, like the vendor update kernel.
        if conv_states is None:
            windows = torch.nn.functional.pad(x.float().unsqueeze(-1), (width - 1, 0))
        else:
            windows = torch.cat(
                [conv_states[cache_indices].float(), x.float().unsqueeze(-1)], dim=-1
            )
            valid = cache_indices != pad_slot_id
            if bool(valid.any()):
                conv_states[cache_indices[valid]] = windows[:, :, 1:].to(
                    conv_states.dtype
                )[valid]
        computed = torch.einsum("bdw,wd->bd", windows, weight_f)
        if bias_f is not None:
            computed = computed + bias_f
        compute_f.copy_(computed)
    else:
        start = 0
        for index, length in enumerate(lengths):
            end = start + length
            slot_value = int(cache_indices[index])
            if slot_value == pad_slot_id:
                compute_f[start:end] = x[start:end].float()
                start = end
                continue
            state = conv_states[slot_value]
            if has_initial_state is not None and not bool(has_initial_state[index]):
                state = torch.zeros_like(state)
            # window is (dim, width - 1 + length): the initial state followed
            # by this sequence's tokens transposed into (dim, length).
            window = torch.cat(
                [state.float(), x[start:end].float().transpose(0, 1)], dim=1
            )
            result = torch.zeros(dim, length, device=device)
            for position in range(width):
                result = result + window[:, position : position + length] * weight_f[
                    position
                ].reshape(-1, 1)
            if bias_f is not None:
                result = result + bias_f.reshape(-1, 1)
            compute_f[start:end] = result.transpose(0, 1)
            conv_states[slot_value] = window[:, -(width - 1) :].to(conv_states.dtype)
            start = end

    return _maybe_silu(compute_f.to(x.dtype), activation_mode)


_CAUSAL_CONV1D_SCHEMA = (
    "npu::causal_conv1d(Tensor x, Tensor weight, Tensor(a!) conv_states, "
    "Tensor? bias=None, Tensor? query_start_loc=None, Tensor? cache_indices=None, "
    "Tensor? has_initial_state=None, Tensor? num_accepted_tokens=None, "
    "int activation_mode=0, int pad_slot_id=-1, int run_mode=0) -> Tensor"
)


def register_causal_conv1d() -> bool:
    """Register ``npu::causal_conv1d`` when no wheel registered it.

    The GDN backend launches ``torch.ops.npu.causal_conv1d`` directly, which
    only exists once the vendor wheel's native extension registers it. This
    defines the same schema (the CANN 9.2.x ABI, with ``conv_states`` marked
    mutated) backed by the plain-PyTorch implementation above. Returns
    whether the registration happened; an existing registration always wins.
    """
    global _CAUSAL_CONV1D_LIBRARY
    try:
        # Resolving through the lazy namespace raises AttributeError while
        # the op is unregistered (and caches the module once it exists).
        torch.ops.npu.causal_conv1d
        return False
    except AttributeError:
        pass
    # The library object owns the registration: it must outlive the process,
    # so it is held at module level (a collected Library also unregisters,
    # which is also where torch's GC-time cache warnings come from).
    _CAUSAL_CONV1D_LIBRARY = torch.library.Library("npu", "FRAGMENT")
    # Library.define takes the op name WITHOUT the namespace prefix (unlike
    # torch.library.define, which takes the full qualified schema).
    _CAUSAL_CONV1D_LIBRARY.define(_CAUSAL_CONV1D_SCHEMA.replace("npu::", "", 1))
    _CAUSAL_CONV1D_LIBRARY.impl(
        "causal_conv1d", _causal_conv1d, "CompositeExplicitAutograd"
    )
    return True


_CAUSAL_CONV1D_LIBRARY: Optional[torch.library.Library] = None


# Placeholder module path -> {attribute: implementation}. Consulted by the
# import guard when it creates each placeholder module.
SGL_KERNEL_NPU_FALLBACKS: dict[str, dict[str, object]] = {
    "sgl_kernel_npu.norm.l1_norm": {"l1_norm": l1_norm},
    "sgl_kernel_npu.fla.fused_gdn_gating": {
        "fused_gdn_gating_npu": fused_gdn_gating_npu,
    },
    "sgl_kernel_npu.fla.chunk": {
        "chunk_gated_delta_rule_npu": chunk_gated_delta_rule_npu,
    },
    "sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent": {
        "fused_sigmoid_gating_delta_rule_update_npu": (
            fused_sigmoid_gating_delta_rule_update_npu
        ),
    },
    "sgl_kernel_npu.mamba.mamba_state_update_triton": {
        "move_cache_dynamic_last_kernel_h_block": move_cache_dynamic_last_kernel_h_block,
        "move_intermediate_cache": move_intermediate_cache,
    },
}
