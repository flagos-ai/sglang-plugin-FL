# SPDX-License-Identifier: Apache-2.0
from transformers.configuration_utils import PreTrainedConfig


class HYV4Config(PreTrainedConfig):
    model_type = "hy_v4"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "n_routed_experts"}

    def __init__(
        self,
        vocab_size=120832,
        hidden_size=2816,
        intermediate_size=6912,
        moe_intermediate_size=768,
        num_hidden_layers=34,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=262144,
        rms_norm_eps=1e-05,
        n_routed_experts=256,
        n_shared_experts=1,
        num_experts_per_tok=8,
        routed_scaling_factor=2.827,
        norm_topk_prob=True,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        mlp_layer_types=None,
        layer_types=None,
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=16,
        indexer_types=None,
        enable_lm_head_fp32=True,
        enable_ihc=True,
        hc_mult=4,
        hc_magnitude=2.0,
        hc_eps=1e-06,
        gated_mla=True,
        gating_type="elementwise",
        learnable_sink=True,
        learnable_sink_init=0.0,
        swiglu_limit=10.0,
        rope_parameters=None,
        num_nextn_predict_layers=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.enable_lm_head_fp32 = enable_lm_head_fp32
        self.enable_ihc = enable_ihc
        self.hc_mult = hc_mult
        self.hc_magnitude = hc_magnitude
        self.hc_eps = hc_eps
        self.gated_mla = gated_mla
        self.gating_type = gating_type
        self.learnable_sink = learnable_sink
        self.learnable_sink_init = learnable_sink_init
        self.swiglu_limit = swiglu_limit
        self.rope_parameters = (
            rope_parameters
            if rope_parameters is not None
            else {"rope_theta": 10000000.0, "rope_type": "default"}
        )
        self.rope_theta = self.rope_parameters["rope_theta"]
        self.rope_interleave = True
        self.indexer_rope_interleave = True
        self.router_fp32 = True
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = qk_rope_head_dim
        self.mlp_layer_types = (
            mlp_layer_types
            if mlp_layer_types is not None
            else ["dense"] + ["sparse"] * (num_hidden_layers - 1)
        )
        self.layer_types = (
            layer_types
            if layer_types is not None
            else ["deepseek_sparse_attention"] * num_hidden_layers
        )
        self.indexer_types = (
            indexer_types
            if indexer_types is not None
            else [
                "full" if i == 0 or (i - 1) % 4 == 0 else "shared"
                for i in range(num_hidden_layers)
            ]
        )
        self.first_k_dense_replace = 1
        self.moe_layer_freq = 1
        self.scoring_func = "sigmoid"
        self.topk_method = "noaux_tc"
        self.n_group = 1
        self.topk_group = 1
        self._validate_hy_v4()

    def _validate_hy_v4(self):
        fields = {
            "mlp_layer_types": self.mlp_layer_types,
            "layer_types": self.layer_types,
            "indexer_types": self.indexer_types,
        }
        for name, values in fields.items():
            if len(values) != self.num_hidden_layers:
                raise ValueError(
                    f"{name} must contain {self.num_hidden_layers} entries, got {len(values)}"
                )
        if set(self.mlp_layer_types) - {"dense", "sparse"}:
            raise ValueError("mlp_layer_types only supports dense and sparse")
        if not set(self.layer_types) <= {
            "deepseek_sparse_attention",
            "indexed_attention",
        }:
            raise ValueError(
                "HYV4 only supports deepseek_sparse_attention (indexed_attention)"
            )
        if set(self.indexer_types) - {"full", "shared"}:
            raise ValueError("indexer_types only supports full and shared")
        if self.indexer_types[0] != "full":
            raise ValueError("indexer_types must start with a full indexer")
        if not self.enable_ihc or self.hc_mult <= 0:
            raise ValueError("HYV4 requires enabled iHC with hc_mult > 0")
        if not self.gated_mla or self.gating_type != "elementwise":
            raise ValueError("HYV4 requires elementwise gated MLA")
        if not self.learnable_sink:
            raise ValueError("HYV4 requires learnable attention sinks")
        if self.q_lora_rank is None:
            raise ValueError("HYV4 sparse attention requires q_lora_rank")


class HYV4VLConfig(PreTrainedConfig):
    model_type = "hy_v4_vl"
    sub_configs = {"text_config": HYV4Config}
    keys_to_ignore_at_inference = ["past_key_values"]
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    def __init__(
        self, text_config=None, vision_config=None, language_model_only=True, **kwargs
    ):
        metadata_keys = {"model_type", "architectures", "auto_map", "processor_class"}
        serving_keys = {
            "quantization_config",
            "compression_config",
            "language_model_only",
        }
        outer_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in metadata_keys | serving_keys
        }
        if isinstance(text_config, HYV4Config):
            text_architectures = text_config.architectures
        else:
            text_architectures = (text_config or {}).get("architectures")
            text_kwargs = {**kwargs, **(text_config or {})}
            for key in metadata_keys:
                text_kwargs.pop(key, None)
            rope = dict(text_kwargs.pop("rope_scaling", None) or {})
            rope.update(text_kwargs.get("rope_parameters") or {})
            rope.setdefault("rope_theta", 10000000.0)
            rope.setdefault("rope_type", "default")
            rope.setdefault("mrope_interleaved", True)
            rope.setdefault("mrope_section", [12, 10, 10])
            text_kwargs["rope_parameters"] = rope
            text_config = HYV4Config(**text_kwargs)
        if text_architectures is None:
            text_architectures = ["HYV4ForCausalLM"]
        if list(text_architectures) not in (
            ["HYV4ForCausalLM"],
            ["HYV4VLForConditionalGeneration"],
        ):
            raise ValueError(
                f"HYV4 VL text_config.architectures must be ['HYV4ForCausalLM'], ['HYV4VLForConditionalGeneration'] or absent, got {text_architectures!r}"
            )
        text_config.architectures = ["HYV4ForCausalLM"]
        if language_model_only is not True:
            raise ValueError(
                "Hy4-preview draft supports text-only inference; media is unsupported"
            )
        outer_kwargs["language_model_only"] = True
        super().__init__(**outer_kwargs)
        self.text_config = text_config
        for name in (
            "dtype",
            "tie_word_embeddings",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
        ):
            setattr(self, name, getattr(text_config, name, None))
        if self.architectures is None:
            self.architectures = ["HYV4VLForConditionalGeneration"]
        self.vision_config = dict(vision_config or {})

    def __getattr__(self, name):
        text_config = self.__dict__.get("text_config")
        if text_config is None or name.startswith("_"):
            raise AttributeError(name)
        return getattr(text_config, name)


__all__ = ["HYV4Config", "HYV4VLConfig"]
