# SPDX-License-Identifier: Apache-2.0
"""Narrow HY4 NextN config/CT aliases; no runtime or device initialization."""

import copy


def configure_hy4_draft(model_config):
    """Called after the native draft mapper, on ModelConfig's private HF copy."""
    config = model_config.hf_config
    if not model_config.is_draft_model or getattr(config, "model_type", None) not in (
        "hy_v4",
        "hy_v4_vl",
    ):
        return
    if config.architectures not in (
        ["HYV4ForCausalLM"],
        ["HYV4VLForConditionalGeneration"],
        ["HYV4ForCausalLMNextN"],
    ):
        raise ValueError("HY4 draft requires its own target or NextN architecture")
    text = model_config.hf_text_config
    if getattr(text, "num_nextn_predict_layers", None) != 1:
        raise ValueError("HY4 bundled NextN requires exactly one checkpoint MTP layer")
    # Keep the outer VL config/mRoPE metadata; only the draft architecture changes.
    config.architectures = ["HYV4ForCausalLMNextN"]
    config.num_nextn_predict_layers = 1


def configure_hy4_hidden_shapes(model_config):
    """HY4 publishes post-HC-head, post-final-norm H, never flattened HC*H."""
    model_config.spec_hidden_size = model_config.hidden_size
    if model_config.is_draft_model:
        # The one MTP block has ordinary residuals, not HC stream state.
        model_config.hc_hidden_size = None
        if (
            model_config.hf_config.architectures != ["HYV4ForCausalLMNextN"]
            or model_config.num_nextn_predict_layers != 1
        ):
            raise ValueError(
                "HY4 draft runtime layer counts require its single NextN block"
            )
        # These ModelConfig execution counts describe the one actual block.
        # Keep HF/text num_hidden_layers=78 for the checkpoint's MTP remap and
        # native NextN loader; never rewrite the checkpoint depth to one.
        model_config.num_hidden_layers = 1
        model_config.num_attention_layers = 1


def _mtp_alias(pattern):
    if not isinstance(pattern, str):
        return None
    if pattern.startswith("re:"):
        # This checkpoint's regex targets only the decoder's experts/indexer/gate.
        # Do not rewrite unrelated model-wide regexes.
        old = r"mtp_layers\.0\."
        if old in pattern:
            return pattern.replace(old, r"decoder\.")
        return None
    old = "model.mtp_layers.0."
    if old not in pattern:
        return None
    tail = pattern.split(old, 1)[1]
    if tail in ("eh_proj", "enorm", "hnorm") or tail.startswith(
        ("eh_proj.", "enorm.", "hnorm.")
    ):
        return "model." + tail
    if tail.startswith("final_layernorm."):
        return "model.shared_head.norm." + tail.removeprefix("final_layernorm.")
    return "model.decoder." + tail


def remap_hy4_mtp_quant_config(quant_config):
    if quant_config is None:
        return None
    if quant_config.get_name() != "compressed_tensors":
        raise ValueError(
            "This HY4 NextN adapter is qualified only for W8A8 compressed-tensors"
        )
    # Native CT resolves both explicit linear targets and regex expert/ignore rules.
    # Copy before appending aliases: target and draft share no mutable scheme map.
    result = copy.deepcopy(quant_config)
    for attribute in ("target_scheme_map", "sparsity_scheme_map"):
        mapping = getattr(result, attribute, None)
        if mapping is None:
            continue
        for pattern, scheme in tuple(mapping.items()):
            alias = _mtp_alias(pattern)
            if alias is not None:
                if alias in mapping and mapping[alias] != scheme:
                    raise ValueError(f"Conflicting HY4 MTP quantization alias: {alias}")
                mapping[alias] = copy.deepcopy(scheme)
    for attribute in ("ignore", "sparsity_ignore_list"):
        patterns = getattr(result, attribute, None)
        if patterns is None:
            continue
        for pattern in tuple(patterns):
            alias = _mtp_alias(pattern)
            if alias is not None and alias not in patterns:
                patterns.append(alias)
    result._hy4_model = True
    return result


def iter_hy4_nextn_weights(weights, target_layers):
    """Keep only real bundled MTP0; delegate sharding/fusion to native loader."""
    original = "model.mtp_layers.0."
    native = f"model.layers.{target_layers}."
    for name, tensor in weights:
        if not name.startswith(original):
            continue
        name = native + name.removeprefix(original)
        if ".final_layernorm." in name:
            name = name.replace(".final_layernorm.", ".shared_head.norm.")
        yield name, tensor
