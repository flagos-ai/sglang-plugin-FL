"""XingChen4 model configuration."""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class XingChen4Config(PretrainedConfig):
    """Transformers-compatible configuration for XingChen4 checkpoints."""

    model_type = "xingchen4"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        architectures: list[str] | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        ep_size: int = 1,
        first_k_dense_replace: int = 0,
        hidden_act: str = "silu",
        hidden_size: int = 3584,
        index_head_dim: int | None = None,
        index_n_heads: int | None = None,
        index_topk: int | None = None,
        initializer_range: float = 0.02,
        intermediate_size: int = 9216,
        kv_lora_rank: int = 512,
        max_position_embeddings: int = 262144,
        moe_intermediate_size: int = 1024,
        moe_layer_freq: int = 1,
        n_group: int = 1,
        n_routed_experts: int = 64,
        n_shared_experts: int = 1,
        norm_topk_prob: bool = True,
        num_attention_heads: int = 32,
        num_experts_per_tok: int = 4,
        num_hidden_layers: int = 40,
        num_key_value_heads: int = 32,
        q_lora_rank: int = 768,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        rms_norm_eps: float = 1e-6,
        rope_scaling: dict[str, Any] | None = None,
        rope_interleave: bool = True,
        rope_theta: float = 10000,
        routed_scaling_factor: float = 2.0,
        scoring_func: str = "sigmoid",
        tie_word_embeddings: bool = False,
        topk_group: int = 1,
        topk_method: str = "noaux_tc",
        use_cache: bool = True,
        v_head_dim: int = 128,
        vocab_size: int = 131072,
        num_residual_streams: int = 4,
        mhc_sinkhorn_iterations: int = 20,
        mhc_init_gating_factor: float = 0.01,
        mhc_h_res_clamp_min: float = -30.0,
        mhc_h_res_clamp_max: float = 30.0,
        num_nextn_predict_layers: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            architectures=architectures or ["XingChen4ForCausalLM"],
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.ep_size = ep_size
        self.first_k_dense_replace = first_k_dense_replace
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.kv_lora_rank = kv_lora_rank
        self.max_position_embeddings = max_position_embeddings
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_layer_freq = moe_layer_freq
        self.n_group = n_group
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.norm_topk_prob = norm_topk_prob
        self.num_attention_heads = num_attention_heads
        self.num_experts_per_tok = num_experts_per_tok
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_scaling = dict(rope_scaling or {})
        self.rope_interleave = rope_interleave
        self.rope_theta = rope_theta
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self.topk_group = topk_group
        self.topk_method = topk_method
        self.use_cache = use_cache
        self.v_head_dim = v_head_dim
        self.vocab_size = vocab_size
        self.num_residual_streams = num_residual_streams
        self.mhc_sinkhorn_iterations = mhc_sinkhorn_iterations
        self.mhc_init_gating_factor = mhc_init_gating_factor
        self.mhc_h_res_clamp_min = mhc_h_res_clamp_min
        self.mhc_h_res_clamp_max = mhc_h_res_clamp_max
        self.num_nextn_predict_layers = num_nextn_predict_layers


class Xing4_0Config(XingChen4Config):
    """Configuration alias used by Xing4.0 checkpoints."""

    model_type = "xing4_0"

    def __init__(self, architectures: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(
            architectures=architectures or ["Xing4_0ForCausalLM"],
            **kwargs,
        )
