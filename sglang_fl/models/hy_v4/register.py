# SPDX-License-Identifier: Apache-2.0
"""HYV4 registration and narrowly scoped text/NextN compatibility hooks."""

import sys

_registered = False


def _hy4(config):
    return getattr(config, "model_type", None) in ("hy_v4", "hy_v4_vl")


def _function_patches(module, name, replacement):
    previous = getattr(module, name)
    patches = [(module, name, previous, replacement)]
    for loaded in tuple(sys.modules.values()):
        if loaded is None or not getattr(loaded, "__name__", "").startswith("sglang."):
            continue
        for key, value in tuple(vars(loaded).items()):
            if value is previous:
                patches.append((loaded, key, previous, replacement))

    return patches


def register_hy_v4():
    global _registered
    if _registered:
        return
    from sglang.srt.configs import model_config as mc
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
    from sglang.srt.models.registry import ModelRegistry
    from transformers import AutoConfig
    from .config import HYV4Config, HYV4VLConfig
    from .model import HYV4ForCausalLM
    from .nextn import HYV4ForCausalLMNextN
    from .nextn_utils import configure_hy4_draft, configure_hy4_hidden_shapes
    from .vl_model import HYV4VLForConditionalGeneration

    previous_draft = mc.ModelConfig._config_draft_model

    def config_draft(self):
        previous_draft(self)
        configure_hy4_draft(self)

    previous_dsa = mc.is_deepseek_dsa
    previous_skip = mc.dsa_layer_skips_topk

    def is_dsa(config):
        return _hy4(config) or previous_dsa(config)

    def skips_topk(config, layer_id):
        if _hy4(config):
            return (
                0 <= layer_id < config.num_hidden_layers
                and config.indexer_types[layer_id] == "shared"
            )
        return previous_skip(config, layer_id)

    previous_shapes = mc.ModelConfig._derive_model_shapes

    def derive_shapes(self):
        previous_shapes(self)
        if _hy4(self.hf_config):
            config = self.hf_text_config
            self.attention_arch = mc.AttentionArch.MLA
            self.head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
            for key in (
                "kv_lora_rank",
                "qk_nope_head_dim",
                "qk_rope_head_dim",
                "v_head_dim",
                "index_head_dim",
            ):
                setattr(self, key, getattr(config, key))
            self._init_mla_scaling(config.rope_parameters)
            configure_hy4_hidden_shapes(self)

    previous_backend = ATTENTION_BACKENDS["dsa"]

    def create_backend(runner):
        if _hy4(runner.model_config.hf_config):
            from .attention_backend import HYV4SparseAttnBackend

            return HYV4SparseAttnBackend(runner)
        return previous_backend(runner)

    # Construct the sink factory and causal wrapper together. The latter calls
    # the former, while the published slot has only one final replacement.
    from .draft_attention_backend import prepare_patches as sink_patches
    from .draft_causal_lengths import prepare_patches as causal_patches

    sinks = sink_patches()
    causal = causal_patches(sinks[0][3])
    causal[0] = (*sinks[0][:3], causal[0][3])
    patches = [
        (mc.ModelConfig, "_config_draft_model", previous_draft, config_draft),
        (mc.ModelConfig, "_derive_model_shapes", previous_shapes, derive_shapes),
        *_function_patches(mc, "is_deepseek_dsa", is_dsa),
        *_function_patches(mc, "dsa_layer_skips_topk", skips_topk),
        sinks[1],
        *causal,
    ]
    # Alias discovery can encounter the defining slot twice.
    unique = {
        (id(owner), name): (owner, name, old, new) for owner, name, old, new in patches
    }
    patches = list(unique.values())
    for owner, name, old, _ in patches:
        if getattr(owner, name) is not old:
            raise RuntimeError(f"HYV4 compatibility owner changed: {name}")

    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    configs = {"hy_v4": HYV4Config, "hy_v4_vl": HYV4VLConfig}
    models = {
        "HYV4ForCausalLM": HYV4ForCausalLM,
        "HYV4VLForConditionalGeneration": HYV4VLForConditionalGeneration,
        "HYV4ForCausalLMNextN": HYV4ForCausalLMNextN,
    }
    missing = object()
    old_configs = {
        key: CONFIG_MAPPING._extra_content.get(key, missing) for key in configs
    }
    old_models = {key: ModelRegistry.models.get(key, missing) for key in models}
    applied = []
    try:
        for key, cls in configs.items():
            AutoConfig.register(key, cls, exist_ok=True)
        ModelRegistry.models.update(models)
        for owner, name, old, new in patches:
            setattr(owner, name, new)
            applied.append((owner, name, old))
        ATTENTION_BACKENDS["dsa"] = create_backend
    except BaseException:
        ATTENTION_BACKENDS["dsa"] = previous_backend
        for owner, name, old in reversed(applied):
            setattr(owner, name, old)
        for mapping, originals in (
            (ModelRegistry.models, old_models),
            (CONFIG_MAPPING._extra_content, old_configs),
        ):
            for key, old in originals.items():
                if old is missing:
                    mapping.pop(key, None)
                else:
                    mapping[key] = old
        raise
    _registered = True
