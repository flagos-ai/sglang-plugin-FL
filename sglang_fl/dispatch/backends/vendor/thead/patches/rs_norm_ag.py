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

"""FlagGems ring RS + norm + AG kernels on the pure TP prefill path of DeepseekV2Model.

SGLANG_FL_RS_NORM_AG=1 turns it on (default off); needs a flag_gems with
fused/rs_norm_ag.py. Everything is AROUND hooks, sglang itself is not modified.

While a forward is active, the residual stays as this rank's rows row_range(T, W, rank)
between layers and the MLP / MoE output stays unreduced (use_reduce_scatter):
  attn -> mlp      fuse1  RS + add + post_attention_layernorm + AG          (every layer)
  mlp  -> attn     scattered layer:  fuse2  RS + add + input_layernorm, my rows only
                                     -> qkv_a GEMM on my rows
                                     -> fuse3  q_a / kv_a norm + AG of the qkv latent
                   other layers:     fuse1 with input_layernorm (full hidden)
  last layer       fuse1 with the final norm, the residual is dropped
A layer is scattered when its attention runs MLA absorb without the DSA indexer (the
indexer needs the full hidden states) and every rank owns at least one row.

Hooks:
  DeepseekV2Model.forward                                  begin / final norm / end
  LayerCommunicator.prepare_attn, .prepare_mlp             the three fused kernels
  LayerCommunicator.should_use_reduce_scatter              True while active
  LayerCommunicator.should_fuse_mlp_allreduce_with_next_layer   False while active

Eager only, one stream, every TP rank must make the same calls: begin() and
_layer_scattered() only look at values that are equal on all ranks.
"""

import logging
import os

import torch
import torch.distributed as dist
from torch import nn

logger = logging.getLogger(__name__)

ENV = "SGLANG_FL_RS_NORM_AG"
_MODEL = "sglang.srt.models.deepseek_v2.DeepseekV2Model"
_COMMUNICATOR = "sglang.srt.layers.communicator.LayerCommunicator"

_ALIGN = 512  # bulk ops need 512B aligned bases

_INITED = False
_COMM = None  # fused_rs_add_rms_norm_ag on the hidden size
_QKV = None  # fused_qkv_norm_ag on the qkv_a output
_ROW_RANGE = None
_STATIC_OK = None

_ACTIVE = False
_T = 0
_B = 0
_E = 0
_SCATTER_OK = False
_NORM_OVERRIDES = []  # (module, previous instance forward or _MISSING)
_MISSING = object()


def _log_rank0(msg: str) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(msg)


def _phys_device_id(index: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return index
    return int(visible.split(",")[index])


def _all_ok(ok: bool, cpu_group) -> bool:
    # every rank must agree, otherwise one rank runs the kernel while another runs pccl
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=cpu_group)
    return int(flag.item()) == 1


def _max_tokens() -> int:
    from sglang.srt.server_args import get_global_server_args

    chunk = get_global_server_args().chunked_prefill_size
    return chunk if chunk and chunk > 0 else 8192


def _plain_rms_norm(norm) -> bool:
    from sglang.srt.layers.layernorm import RMSNorm

    return (
        isinstance(norm, RMSNorm)
        and norm.variance_size_override is None
        and not norm.cast_x_before_out_mul
        and not norm.fp32_residual
        and norm.weight.dtype == torch.bfloat16
    )


def _static_ok(model) -> bool:
    """Server / model properties that do not change between forwards."""
    from sglang.srt.distributed import (
        get_pp_group,
        get_tensor_model_parallel_world_size,
    )
    from sglang.srt.layers.communicator import (
        LayerCommunicator,
        enable_moe_dense_fully_dp,
        get_attn_tp_context,
    )
    from sglang.srt.layers.dp_attention import (
        get_attention_tp_size,
        is_dp_attention_enabled,
        is_enable_moe_cp_allgather,
    )
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.server_args import get_global_server_args

    args = get_global_server_args()
    tp = get_tensor_model_parallel_world_size()
    if not (
        tp > 1
        and get_attention_tp_size() == tp
        and not is_dp_attention_enabled()
        and get_pp_group().world_size == 1
        and get_moe_a2a_backend().is_none()
        and not enable_moe_dense_fully_dp()
        and not is_enable_moe_cp_allgather()
        and not get_attn_tp_context().allow_input_scattered
        and args.disable_piecewise_cuda_graph
        and not getattr(model, "dsa_enable_prefill_cp", False)
        and not getattr(model, "mla_enable_prefill_cp", False)
        and not model.layers_to_capture
        and _plain_rms_norm(model.norm)
    ):
        return False
    for layer in model.layers[model.start_layer : model.end_layer]:
        comm = getattr(layer, "layer_communicator", None)
        if type(comm) is not LayerCommunicator or not comm.allow_reduce_scatter:
            return False
        if not (
            _plain_rms_norm(layer.input_layernorm)
            and _plain_rms_norm(layer.post_attention_layernorm)
        ):
            return False
        if getattr(layer.mlp, "_shared_expert_tp1", False):
            return False
    return True


def _init(model) -> None:
    """Create both communicators on the TP group. Runs collectives: every rank calls it
    at the same point (the first eligible forward). Failures disable it everywhere."""
    global _INITED, _COMM, _QKV, _ROW_RANGE
    _INITED = True

    from sglang.srt.distributed import get_tp_group

    group = get_tp_group()
    cpu_group = group.cpu_group
    attn = model.layers[model.start_layer].self_attn
    H = model.norm.weight.numel()
    q_dim = getattr(attn, "q_lora_rank", None)
    mod, ok = None, H % 256 == 0
    if ok:
        try:
            from flag_gems.fused import rs_norm_ag as mod
        except Exception as e:
            logger.warning(f"FlagGems rs-norm-ag import failed: {e}")
            ok = False
    if not _all_ok(ok, cpu_group):
        _log_rank0("FlagGems rs-norm-ag: disabled (import / hidden size)")
        return

    phys = [None] * group.world_size
    dist.all_gather_object(phys, _phys_device_id(group.device.index), group=cpu_group)
    rings = mod.ppu810e_rings(phys)
    max_tokens = _max_tokens()
    comm = qkv = None
    try:
        comm = mod.fused_rs_add_rms_norm_ag(
            cpu_group,
            group.device,
            max_tokens,
            H,
            torch.bfloat16,
            cpu_group=cpu_group,
            rings=rings,
        )
        if q_dim is not None:
            kv_dim = attn.kv_lora_rank
            qkv = mod.fused_qkv_norm_ag(
                cpu_group,
                group.device,
                max_tokens,
                q_dim,
                kv_dim,
                q_dim + kv_dim + attn.qk_rope_head_dim,
                torch.bfloat16,
                cpu_group=cpu_group,
                rings=rings,
            )
        ok = True
    except Exception as e:
        logger.warning(f"FlagGems rs-norm-ag init failed: {e}")
        ok = False
    if _all_ok(ok, cpu_group):
        _COMM, _QKV, _ROW_RANGE = comm, qkv, mod.row_range
    else:
        for c in (comm, qkv):
            if c is not None:
                c.close()
    _log_rank0(
        f"FlagGems rs-norm-ag: "
        f"{f'enabled, PPUs {phys}, max_tokens {max_tokens}, qkv {_QKV is not None}' if _COMM else 'disabled (init failed)'}"
    )


def _begin(model, num_tokens: int, forward_batch) -> bool:
    """Decide for this forward (same answer on every rank); called before the layers."""
    global _ACTIVE, _T, _B, _E, _SCATTER_OK, _STATIC_OK

    _ACTIVE = False
    mode = forward_batch.forward_mode
    if not (
        mode.is_extend()
        and not mode.is_target_verify()
        and not mode.is_draft_extend()
        and not forward_batch.can_run_tbo
        and forward_batch.input_ids is not None
    ):
        return False
    if torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling():
        return False
    T = num_tokens
    if T == 0:
        return False
    if _STATIC_OK is None:
        _STATIC_OK = _static_ok(model)
    if not _STATIC_OK:
        return False
    if not _INITED:
        _init(model)
    if _COMM is None or T > _COMM.max_tokens:
        return False

    W = _COMM.world_size
    _T = T
    _B, _E = _ROW_RANGE(T, W, _COMM.rank)
    # the last rank owns rows too: no rank runs the qkv_a GEMM on 0 rows
    _SCATTER_OK = _QKV is not None and (W - 1) * ((T + W - 1) // W) < T
    _ACTIVE = True
    return True


def _restore_norms() -> None:
    while _NORM_OVERRIDES:
        m, prev = _NORM_OVERRIDES.pop()
        if prev is _MISSING:
            m.__dict__.pop("forward", None)
        else:
            m.__dict__["forward"] = prev


def _end() -> None:
    global _ACTIVE
    _restore_norms()
    _ACTIVE = False


def _identity_norm(x, *args, **kwargs):
    # q_a / kv_a norm already done by fuse3; forward_mla still calls them
    return x.contiguous()


def _skip_norm(m) -> None:
    _NORM_OVERRIDES.append((m, m.__dict__.get("forward", _MISSING)))
    m.__dict__["forward"] = _identity_norm


def _layer_scattered(attn, qkv_latent_func, forward_batch) -> bool:
    """fuse2 + fuse3 for this layer: MLA absorb without the DSA indexer."""
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
        AttnForwardMethod,
    )

    if not _SCATTER_OK or attn is None or qkv_latent_func is None:
        return False
    if getattr(attn, "q_lora_rank", None) is None:
        return False
    if getattr(attn, "use_dsa", False) and not attn.skip_topk:
        return False
    if not (
        _plain_rms_norm(attn.q_a_layernorm) and _plain_rms_norm(attn.kv_a_layernorm)
    ):
        return False
    return attn.dispatch_attn_forward_method(forward_batch) == AttnForwardMethod.MLA


def _aligned(t: torch.Tensor) -> torch.Tensor:
    if t.is_contiguous() and (t.numel() == 0 or t.data_ptr() % _ALIGN == 0):
        return t
    out = torch.empty_like(t, memory_format=torch.contiguous_format)
    assert out.numel() == 0 or out.data_ptr() % _ALIGN == 0, (
        "allocator returned a misaligned block"
    )
    out.copy_(t)
    return out


def _fuse(x: torch.Tensor, residual: torch.Tensor, norm, gather: bool) -> torch.Tensor:
    """residual: this rank's rows, updated in place."""
    return _COMM(
        _aligned(x),
        residual,
        _aligned(norm.weight.data),
        norm.variance_epsilon,
        gather=gather,
    )


# ---- hooks ----


class _FinalNorm(nn.Module):
    """Stands in for DeepseekV2Model.norm during an active forward: the last layer's MLP
    partial sums + the scattered residual -> full normed hidden (fuse1)."""

    def __init__(self, norm):
        super().__init__()
        self.__dict__["real"] = norm  # not a submodule: stays out of state_dict

    def forward(self, hidden_states, residual=None):
        if residual is None or not _ACTIVE:
            return (
                self.real(hidden_states, residual)
                if residual is not None
                else self.real(hidden_states)
            )
        return _fuse(hidden_states, residual, self.real, gather=True), None


def _model_forward_hook(
    original_fn, self, input_ids, positions, forward_batch, *args, **kwargs
):
    input_embeds = kwargs.get("input_embeds", args[0] if args else None)
    x = input_embeds if input_embeds is not None else input_ids
    if x is None or not _begin(self, x.shape[0], forward_batch):
        return original_fn(self, input_ids, positions, forward_batch, *args, **kwargs)
    real = self._modules["norm"]
    self._modules["norm"] = _FinalNorm(real)
    try:
        return original_fn(self, input_ids, positions, forward_batch, *args, **kwargs)
    finally:
        self._modules["norm"] = real
        _end()


def _prepare_attn_hook(
    original_fn, self, hidden_states, residual, forward_batch, *args, **kwargs
):
    from sglang.srt.layers.communicator import (
        AttentionInputs,
        LayerCommunicator,
        get_attn_tp_context,
    )

    if not _ACTIVE or type(self) is not LayerCommunicator:
        return original_fn(
            self, hidden_states, residual, forward_batch, *args, **kwargs
        )
    post_residual_addition = kwargs.get(
        "post_residual_addition", args[1] if len(args) > 1 else None
    )
    assert post_residual_addition is None, (
        "rs-norm-ag: post_residual_addition unsupported"
    )

    func = self.qkv_latent_func
    attn = getattr(func, "__self__", None)
    scattered = _layer_scattered(attn, func, forward_batch)
    if residual is None:
        # layer 0: the embedding is replicated, keep my rows as the residual
        residual = _aligned(hidden_states[_B:_E].clone())
        hidden_states = self.input_layernorm(hidden_states)
        if scattered:
            hidden_states = hidden_states[_B:_E]
    else:
        hidden_states = _fuse(
            hidden_states, residual, self.input_layernorm, gather=not scattered
        )

    if func is not None:
        inputs = AttentionInputs(hidden_states, forward_batch, func)
        if scattered:
            qn, kvn = attn.q_a_layernorm, attn.kv_a_layernorm
            # fuse3 now; fetch_qkv_latent returns the cached full latent
            inputs.qkv_latent_ = _QKV(
                func(hidden_states, forward_batch).contiguous(),
                _T,
                qn.weight.data,
                qn.variance_epsilon,
                kvn.weight.data,
                kvn.variance_epsilon,
            )
            _skip_norm(qn)
            _skip_norm(kvn)
        get_attn_tp_context().set_attn_inputs(inputs)
    return hidden_states, residual


def _prepare_mlp_hook(
    original_fn, self, hidden_states, residual, forward_batch, *args, **kwargs
):
    from sglang.srt.layers.communicator import LayerCommunicator

    if not _ACTIVE or type(self) is not LayerCommunicator:
        return original_fn(
            self, hidden_states, residual, forward_batch, *args, **kwargs
        )
    _restore_norms()
    # o_proj partial sums -> full normed hidden
    return _fuse(
        hidden_states, residual, self.post_attention_layernorm, gather=True
    ), residual


def _use_reduce_scatter_hook(original_fn, self, *args, **kwargs):
    # the next layer's fuse2 / the final norm's fuse1 does the reduction
    return True if _ACTIVE else original_fn(self, *args, **kwargs)


def _fuse_mlp_allreduce_hook(original_fn, self, *args, **kwargs):
    return False if _ACTIVE else original_fn(self, *args, **kwargs)


def patch_rs_norm_ag() -> bool:
    """Register the hooks when SGLANG_FL_RS_NORM_AG=1. Returns whether it did."""
    if os.environ.get(ENV, "0").strip().lower() not in ("1", "true", "yes", "on"):
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(f"{_MODEL}.forward", _model_forward_hook, HookType.AROUND)
    HookRegistry.register(
        f"{_COMMUNICATOR}.prepare_attn", _prepare_attn_hook, HookType.AROUND
    )
    HookRegistry.register(
        f"{_COMMUNICATOR}.prepare_mlp", _prepare_mlp_hook, HookType.AROUND
    )
    HookRegistry.register(
        f"{_COMMUNICATOR}.should_use_reduce_scatter",
        _use_reduce_scatter_hook,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{_COMMUNICATOR}.should_fuse_mlp_allreduce_with_next_layer",
        _fuse_mlp_allreduce_hook,
        HookType.AROUND,
    )
    _log_rank0(f"FlagGems rs-norm-ag hooks registered ({ENV}=1)")
    return True
