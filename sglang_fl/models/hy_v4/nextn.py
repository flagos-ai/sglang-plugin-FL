# SPDX-License-Identifier: Apache-2.0
"""HY4 W8A8 bundled NextN on the pinned SGLang 0.5.18 runtime.

One ordinary-residual HY4 block, genuine INT8 eh_proj/expert/attention weights,
and the target's shared embedding/head. No DSV3 architecture alias or HC block.
"""

import torch
from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
from sglang.srt.layers.communicator import AttentionInputs
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    get_embedding_tp_kwargs,
)
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.srt.runtime_context import get_forward, get_parallel, get_stream
from sglang.srt.utils import BumpAllocator, add_prefix, is_cuda
from torch import nn

from .model import (
    HYV4_PACKED_MODULES_MAPPING,
    HYV4Attention,
    HYV4ForCausalLM,
    hyv4_mrope_rows_are_equal,
    normalize_hyv4_weight_scale,
    permute_hyv4_indexer_weight,
)
from .nextn_utils import iter_hy4_nextn_weights, remap_hy4_mtp_quant_config


class HYV4MTPDecoderLayer(nn.Module):
    def __init__(
        self, config, quant_config=None, prefix="", alt_stream=None, *, use_mrope=False
    ):
        super().__init__()
        self.self_attn = HYV4Attention(
            config,
            0,
            quant_config,
            add_prefix("self_attn", prefix),
            alt_stream,
            is_nextn=True,
            use_mrope=use_mrope,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = DeepseekV2MoE(
            config,
            0,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
            alt_stream=alt_stream,
            is_nextn=True,
        )
        if hasattr(self.mlp, "shared_experts"):
            self.mlp.shared_experts.swiglu_limit = None

    def forward(
        self,
        positions,
        hidden_states,
        forward_batch,
        zero_allocator,
        prev_topk_indices=None,
    ):
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        inputs = AttentionInputs(
            normalized, forward_batch, self.self_attn.prepare_qkv_latent
        )
        with get_forward().scoped(attn_inputs=inputs):
            output = self.self_attn(
                positions,
                normalized,
                forward_batch,
                zero_allocator,
                prev_topk_indices=prev_topk_indices,
            )
        if isinstance(output, tuple):
            output, topk_indices = output
        else:
            topk_indices = None
        # HYV4Attention owns a reducing o_proj. The pinned MoE forward owns
        # its routed/shared-expert reduction. Adding a second collective is wrong.
        hidden_states = residual + output
        residual = hidden_states
        output = self.mlp(self.post_attention_layernorm(hidden_states), forward_batch)
        return residual + output, topk_indices


class HYV4ModelNextN(nn.Module):
    def __init__(self, config, quant_config=None, prefix="", *, use_mrope=False):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
            **get_embedding_tp_kwargs(),
        )
        self.enorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            2 * config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("eh_proj", prefix),
        )
        self.alt_stream = get_stream("alt") if is_cuda() else None
        self.decoder = HYV4MTPDecoderLayer(
            config,
            quant_config,
            add_prefix("decoder", prefix),
            self.alt_stream,
            use_mrope=use_mrope,
        )
        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    @property
    def layers(self):
        # Native prefill graph resolver requires .layers. A read-only view
        # avoids registering the same module twice or changing weight names.
        return (self.decoder,)

    def embed_input_ids(self, input_ids, forward_batch):
        if (
            forward_batch.forward_mode.is_extend()
            and forward_batch.contains_mm_inputs()
        ):
            embeds = forward_batch.mm_input_embeds
            if embeds is None:
                raise ValueError(
                    "HY4 NextN multimodal prefill requires shifted target embeddings"
                )
            last = (
                forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
            ).long()
            embeds[last] = self.embed_tokens(input_ids[last])
            return embeds
        return self.embed_tokens(input_ids)

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        hidden = (
            self.embed_input_ids(input_ids, forward_batch)
            if input_embeds is None
            else input_embeds
        )
        if hidden.shape[0] > 0:
            hidden = self.eh_proj(
                torch.cat(
                    (
                        self.enorm(hidden),
                        self.hnorm(forward_batch.spec_info.hidden_states),
                    ),
                    dim=-1,
                )
            )[0]
        zeros = BumpAllocator(buffer_size=2, dtype=torch.float32, device=hidden.device)
        carry = IndexTopKShareState.from_mtp_carry(forward_batch)
        hidden, topk = self.decoder(
            positions, hidden, forward_batch, zeros, carry.topk_indices
        )
        carry.update(topk)
        carry.publish()
        if not forward_batch.forward_mode.is_idle():
            hidden = self.shared_head.norm(hidden)
        return hidden


class HYV4ForCausalLMNextN(nn.Module, DeepseekV2WeightLoaderMixin):
    packed_modules_mapping = HYV4_PACKED_MODULES_MAPPING

    @classmethod
    def shared_experts_fusion_disable_reason(cls, hf_config, quant_config):
        return HYV4ForCausalLM.shared_experts_fusion_disable_reason(
            getattr(hf_config, "text_config", hf_config), quant_config
        )

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        if quant_config is None:
            raise ValueError(
                "HY4 W8A8 NextN requires its compressed-tensors weight config"
            )
        self.config = getattr(config, "text_config", config)
        if getattr(self.config, "num_nextn_predict_layers", None) != 1:
            raise ValueError("HY4 NextN requires exactly one bundled draft block")
        self.pp_group = get_parallel().pp_group
        if self.pp_group.world_size != 1:
            raise ValueError("This HY4 W8A8 NextN adapter is limited to PP1")
        self.is_mrope_enabled = (
            "mrope_section" in self.config.rope_parameters
            and not getattr(config, "language_model_only", False)
        )
        self.quant_config = remap_hy4_mtp_quant_config(quant_config)
        if self.quant_config is not None:
            from .w8a8_moe import install_w8a8_moe

            install_w8a8_moe()
        self.model = HYV4ModelNextN(
            self.config,
            self.quant_config,
            add_prefix("model", prefix),
            use_mrope=self.is_mrope_enabled,
        )
        self.num_fused_shared_experts = self.model.decoder.mlp.num_fused_shared_experts
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=self.quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(self.config)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch):
        if self.is_mrope_enabled and not forward_batch.forward_mode.is_idle():
            positions = forward_batch.mrope_positions
            if hyv4_mrope_rows_are_equal(forward_batch.forward_mode):
                positions = positions[0]
        hidden = self.model(input_ids, positions, forward_batch)
        # Intentionally publish post-final-norm H, matching the target HY4 and
        # reference NextN. Never substitute HC*H streams or pre-final-norm state.
        return self.logits_processor(input_ids, hidden, self.lm_head, forward_batch)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head

    def load_weights(self, weights):
        def mapped():
            for name, tensor in iter_hy4_nextn_weights(
                weights, self.config.num_hidden_layers
            ):
                name, tensor = normalize_hyv4_weight_scale(
                    name, tensor, self.quant_config
                )
                yield name, permute_hyv4_indexer_weight(name, tensor, self.config)

        self.do_load_weights(mapped(), is_nextn=True)

    def post_load_weights(self, is_nextn=True, weight_names=None):
        if not is_nextn:
            raise ValueError("HY4 NextN cannot load the target layer stack")
        super().post_load_weights(is_nextn=True, weight_names=weight_names)
        attn = self.model.decoder.self_attn
        if attn.w_kc is not None and attn.w_kc.dtype == torch.int8:
            weight, scale = attn.kv_b_proj.weight, attn.kv_b_proj.weight_scale
            if scale.numel() != weight.shape[0]:
                raise ValueError("HY4 NextN INT8 MLA requires channel scales")
            weight = weight.to(torch.bfloat16) * scale.to(torch.bfloat16).reshape(-1, 1)
            key, value = weight.unflatten(
                0, (-1, attn.qk_nope_head_dim + attn.v_head_dim)
            ).split([attn.qk_nope_head_dim, attn.v_head_dim], dim=1)
            attn.w_kc = key.transpose(1, 2).contiguous().transpose(1, 2)
            attn.w_vc = value.contiguous().transpose(1, 2)


EntryClass = [HYV4ForCausalLMNextN]
