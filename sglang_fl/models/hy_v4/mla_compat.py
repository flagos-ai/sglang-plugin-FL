# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    per_tensor_quant_mla_fp8,
    per_token_group_quant_mla_deep_gemm_masked_fp8,
)
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.dcp import (
    cp_lse_ag_out_rs_mla,
    dcp_a2a_lse_reduce,
)
from sglang.srt.layers.logits_processor import get_in_autotune_dummy_run
from sglang.srt.lora.deepseek_mla_correction import (
    apply_v_correction as apply_kv_b_lora_v_correction,
)
from sglang.srt.lora.deepseek_mla_correction import (
    is_kv_b_lora_active,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.models.deepseek_common.utils import (
    FORWARD_ABSORB_CORE_ATTENTION_BACKENDS,
    _is_cpu,
    _is_cublas_ge_129,
    _is_musa,
)
from sglang.srt.runtime_context import get_parallel

from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    MlaBmmFusionPlan,
    mla_bmm_then_unified_attention,
    bcg_mla_bmm_then_unified_attention,
    is_dcp_mla_decode_phase,
    is_mla_dcp_lse_base_on_e,
    _select_local_dcp_heads_for_autotune,
)
from sglang.kernels.ops.gemm import bmm_fp8

logger = logging.getLogger(__name__)
_SGLANG_EXPERIMENTAL_LORA_OPTI = envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()
_ENABLE_DSA_Q8KV8_BORN_FP8_Q = envs.SGLANG_ENABLE_DSA_Q8KV8_BORN_FP8_Q.get()
_ENABLE_DSA_Q8KV8_QPREP_OVERLAP = envs.SGLANG_ENABLE_DSA_Q8KV8_QPREP_OVERLAP.get()

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA


class HYV4MLAForwardMixin:
    def forward_absorb_core(
        self: DeepseekV2AttentionMLA,
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
        llama_4_scaling,
        fusion_plan: Optional[MlaBmmFusionPlan] = None,
        attention_output_gate=None,
    ):
        save_kv_cache = True

        if self.current_attention_backend in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS:
            extra_args = {}
            if self._fuse_rope_for_trtllm_mla(forward_batch):
                extra_args = {
                    "cos_sin_cache": self.rotary_emb.cos_sin_cache,
                    "is_neox": self.rotary_emb.is_neox_style,
                    "llama_4_scaling": llama_4_scaling,
                }
            if fusion_plan is not None:
                bmm_attention_fn = (
                    bcg_mla_bmm_then_unified_attention
                    if is_in_breakable_cuda_graph()
                    else mla_bmm_then_unified_attention
                )
                bmm_attention_fn(
                    fusion_plan.q_nope_t,
                    self.w_kc,
                    fusion_plan.q_nope_out_buf,
                    q_nope_out,
                    k_nope,
                    fusion_plan.attn_output_buf,
                    save_kv_cache,
                    self.layer_id,
                    q_pe,
                    k_pe,
                    cos_sin_cache=extra_args.get("cos_sin_cache"),
                    is_neox=extra_args.get("is_neox"),
                    llama_4_scaling=extra_args.get("llama_4_scaling"),
                    topk_indices=topk_indices,
                )
                attn_output = fusion_plan.attn_output_buf
            elif is_dcp_mla_decode_phase(forward_batch):
                # set return_lse=True to correct attn_output
                attn_output, lse = self.attn_mqa_for_dcp_decode(
                    q_nope_out,
                    k_nope,
                    k_nope,
                    forward_batch,
                    q_rope=q_pe,
                    k_rope=k_pe,
                    **extra_args,
                    **(
                        dict(topk_indices=topk_indices)
                        if topk_indices is not None
                        else {}
                    ),
                )
            else:
                attn_output = self.attn_mqa(
                    q_nope_out,
                    k_nope,
                    k_nope,
                    forward_batch,
                    q_rope=q_pe,
                    k_rope=k_pe,
                    **extra_args,
                    **(
                        dict(topk_indices=topk_indices)
                        if topk_indices is not None
                        else {}
                    ),
                )
        else:
            q = torch.cat([q_nope_out, q_pe], dim=-1)
            k = torch.cat([k_nope, k_pe], dim=-1)

            # Apply llama 4 scaling if provided
            if llama_4_scaling is not None:
                q *= llama_4_scaling

            attn_output = self.attn_mqa(
                q,
                k,
                k_nope,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
            )

        # correct attn_output with respect to lse from other ranks
        if is_dcp_mla_decode_phase(forward_batch):
            attn_output = attn_output.view(
                -1,
                self.num_local_heads * get_parallel().attn_dcp_size,
                self.kv_lora_rank,
            )
            if get_in_autotune_dummy_run():
                # The synthetic FlashInfer MoE autotune pass discards model
                # outputs. Avoid an unnecessary cross-node MNNVL exchange of
                # zero attention partials.
                attn_output = _select_local_dcp_heads_for_autotune(
                    attn_output, self.num_local_heads
                )
            else:
                dcp_comm_backend = get_parallel().dcp_comm_backend
                is_lse_base_on_e = is_mla_dcp_lse_base_on_e(
                    self.current_attention_backend
                )
                if dcp_comm_backend in ("a2a", "fi_a2a"):
                    # A2A exchange of head partials + LSE, then local Triton combine.
                    attn_output = dcp_a2a_lse_reduce(
                        attn_output.contiguous(),
                        lse.contiguous(),
                        get_parallel().dcp_group,
                        is_lse_base_on_e=is_lse_base_on_e,
                        comm_backend=dcp_comm_backend,
                    )
                else:
                    attn_output = cp_lse_ag_out_rs_mla(
                        attn_output,
                        lse,
                        get_parallel().dcp_group,
                        is_lse_base_on_e=is_lse_base_on_e,
                    )
                    attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        _kvb_v = None
        if _SGLANG_EXPERIMENTAL_LORA_OPTI:
            # Fork the kv_b v-correction A-step onto the LoRA side stream to overlap the bmm.
            from sglang.srt.lora.trtllm_lora_temp.deepseek_mla_correction import (
                kv_b_lora_v_prepare,
            )

            _kvb_v = kv_b_lora_v_prepare(self, attn_output)

        if self.use_deep_gemm_bmm:
            (
                attn_output_val,
                attn_output_scale,
                masked_m,
                expected_m,
                aligned_m,
            ) = per_token_group_quant_mla_deep_gemm_masked_fp8(
                attn_output.transpose(0, 1)
            )
            attn_bmm_output = attn_output.new_empty(
                (self.num_local_heads, aligned_m, self.v_head_dim)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (attn_output_val, attn_output_scale),
                (self.w_vc, self.w_scale_v),
                attn_bmm_output,
                masked_m,
                expected_m,
            )
            attn_bmm_output = (
                attn_bmm_output[:, :expected_m, :].transpose(0, 1).flatten(1, 2)
            )
        elif self.w_vc.dtype == torch.float8_e4m3fn:
            if _is_cpu:
                attn_bmm_output = torch.bmm(
                    attn_output.to(torch.bfloat16).transpose(0, 1),
                    self.w_vc.to(torch.bfloat16) * self.w_scale,
                )
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
            else:
                attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                    attn_output.transpose(0, 1),
                    (
                        torch.zeros(
                            (1,), dtype=torch.float32, device=attn_output.device
                        )
                        if _is_cublas_ge_129
                        else zero_allocator.allocate(1)
                    ),
                )
                attn_bmm_output = bmm_fp8(
                    attn_output_val,
                    self.w_vc,
                    attn_output_scale,
                    self.w_scale,
                    torch.bfloat16,
                )
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        elif _is_musa:
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1), self.w_vc
            )
            attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        else:
            if is_in_tc_piecewise_cuda_graph():
                # torch dynamo requires out= op was called where output tensor was non-contiguous
                attn_bmm_output = (
                    torch.bmm(attn_output.transpose(0, 1), self.w_vc)
                    .transpose(0, 1)
                    .flatten(1, 2)
                )
            else:
                attn_bmm_output = torch.empty(
                    (attn_output.shape[0], self.num_local_heads * self.v_head_dim),
                    dtype=attn_output.dtype,
                    device=attn_output.device,
                )
                torch.bmm(
                    attn_output.transpose(0, 1),
                    self.w_vc,
                    out=attn_bmm_output.view(
                        -1, self.num_local_heads, self.v_head_dim
                    ).transpose(0, 1),
                )
        if _SGLANG_EXPERIMENTAL_LORA_OPTI:
            from sglang.srt.lora.trtllm_lora_temp.deepseek_mla_correction import (
                kv_b_lora_v_apply,
            )

            attn_bmm_output = kv_b_lora_v_apply(
                self, attn_output, attn_bmm_output, _kvb_v
            )
        elif is_kv_b_lora_active(self):
            attn_bmm_output = apply_kv_b_lora_v_correction(
                self, attn_output, attn_bmm_output
            )
        attn_bmm_output = self.apply_attention_output_gate(
            attn_bmm_output, attention_output_gate
        )
        output, _ = self.o_proj(attn_bmm_output)

        if self.next_skip_topk is None:
            return output

        # Return topk_indices for the next layer when enabling index cache
        if not self.next_skip_topk:
            return output, None
        else:
            return output, topk_indices
