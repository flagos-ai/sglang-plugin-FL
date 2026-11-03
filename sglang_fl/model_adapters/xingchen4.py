"""Runtime compatibility hooks for XingChen4 and Xing4.0."""

from __future__ import annotations


_XINGCHEN4_MLA_ARCHITECTURES = {
    "XingChen4ForCausalLM",
    "Xing4_0ForCausalLM",
}


def _derive_model_shapes_hook(original_fn, self):
    """Teach SGLang's model config that XingChen4 uses MLA cache layout.

    SGLang 0.5.18 derives ``attention_arch`` from a closed architecture-name
    list.  Out-of-tree model registration alone therefore leaves Xing4 as MHA,
    which allocates an ``[num_kv_heads, head_dim]`` cache while the model writes
    the compressed ``[1, kv_lora_rank + qk_rope_head_dim]`` MLA cache.
    """

    result = original_fn(self)
    architectures = set(getattr(self.hf_config, "architectures", None) or ())
    if not architectures.intersection(_XINGCHEN4_MLA_ARCHITECTURES):
        return result

    from sglang.srt.configs.model_config import AttentionArch

    config = self.hf_text_config
    self.head_dim = 256
    self.attention_arch = AttentionArch.MLA
    self.kv_lora_rank = config.kv_lora_rank
    self.qk_nope_head_dim = config.qk_nope_head_dim
    self.qk_rope_head_dim = config.qk_rope_head_dim
    self.v_head_dim = config.v_head_dim
    self.index_head_dim = None
    self._init_mla_scaling(getattr(config, "rope_scaling", None))
    return result


def register_xingchen4_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from .xingchen4_detokenizer import register_xingchen4_detokenizer_hook

    HookRegistry.register(
        "sglang.srt.configs.model_config.ModelConfig._derive_model_shapes",
        _derive_model_shapes_hook,
        HookType.AROUND,
    )
    register_xingchen4_detokenizer_hook()
