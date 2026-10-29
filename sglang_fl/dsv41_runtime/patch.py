# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Register model-specific hooks before worker initialization and Graph capture."""

import os
from types import FunctionType

_registered = False


class _NativeProtocolView:
    def __init__(self, config):
        self._config = config

    @property
    def disaggregation_transfer_backend(self):
        return "mooncake"

    def __getattr__(self, name):
        return getattr(self._config, name)


def _validate_features(original, server_args):
    # The native validator still checks every model/cache/graph feature. Only
    # its protocol name is viewed as Mooncake; real routing never changes.
    resolve = original.__globals__["resolving_view"]
    cfg = resolve(server_args)
    if cfg.disaggregation_transfer_backend != "flagcx":
        return original(server_args)
    model_config_of = original.__globals__.get("model_config_of")
    model_config = (
        model_config_of(server_args)
        if model_config_of is not None
        else server_args.get_model_config()
    )
    if model_config.hf_config.model_type != "deepseek_v41":
        return original(server_args)
    if cfg.disaggregation_mode != "null" and cfg.speculative_algorithm is not None:
        from .pd import allow_decode_config

        equal = cfg.dp_size == 1 and not cfg.enable_dp_attention
        dp4 = cfg.dp_size == 4 and allow_decode_config(cfg)
        if cfg.tp_size != 8 or cfg.ep_size != 8 or not (equal or dp4):
            raise ValueError(
                "V4.1 FlagCX DSpark PD requires TP8/EP8 with equal attention TP or decode TP2/DP4"
            )

    def protocol_view(args):
        value = resolve(args)
        return (
            _NativeProtocolView(value)
            if value.disaggregation_transfer_backend == "flagcx"
            else value
        )

    validate = FunctionType(
        original.__code__,
        dict(original.__globals__, resolving_view=protocol_view),
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    validate.__kwdefaults__ = original.__kwdefaults__
    return validate(server_args)


def _draft_forward(original, *args, **kwargs):
    # Draft experts have their own fixed placement and no target layer index.
    # Their warmup/capture must not enter the target model's EPLB statistics.
    from sglang.srt.eplb.expert_distribution import (
        get_global_expert_distribution_recorder,
    )

    with get_global_expert_distribution_recorder().disable_this_region():
        return original(*args, **kwargs)


def _register_indexer_hooks(registry, hook_type, producer, compact):
    from sglang_fl import _build_config, _get_oot_filters, _oot_enabled_for, _parse_bool

    if _parse_bool(os.environ.get("SGLANG_FL_OOT_ENABLED", "1"), default=True):
        _, whitelist, blacklist = _get_oot_filters(_build_config())
        backend = "sglang.srt.layers.attention.deepseek_v4_backend"
        for op_name, target, hook in (
            ("dsv41_indexer_producer", "fp4_index_logits_req_to_token", producer),
            ("dsv41_indexer_compact", "fp4_index_logits_candidate_blocks", compact),
        ):
            if _oot_enabled_for(op_name, whitelist, blacklist):
                registry.register(f"{backend}.{target}", hook, hook_type.AROUND)


def apply_dsv41_patches():
    global _registered
    if _registered:
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from sglang_fl.disaggregation.patch import apply_disaggregation_patch

    from .indexer import compact, producer
    from .pd import install_native_layout_hooks
    from .prefill import candidate_block_ids, candidate_blocks, extend_indexer

    apply_disaggregation_patch()
    install_native_layout_hooks()
    HookRegistry.register(
        "sglang.srt.arg_groups.deepseek_v4_hook.validate_deepseek_v41_features",
        _validate_features,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.models.deepseek_v4_dspark.DeepseekV4ForCausalLMDSpark.forward",
        _draft_forward,
        HookType.AROUND,
    )
    _register_indexer_hooks(HookRegistry, HookType, producer, compact)
    indexer = "sglang.srt.layers.attention.dsv4.indexer"
    HookRegistry.register(
        f"{indexer}.select_candidate_blocks", candidate_blocks, HookType.REPLACE
    )
    HookRegistry.register(
        f"{indexer}.select_candidate_block_ids", candidate_block_ids, HookType.REPLACE
    )
    HookRegistry.register(
        "sglang.srt.layers.attention.deepseek_v4_backend."
        "DeepseekV4AttnBackend._low_ratio_index_topk_sm90_extend",
        extend_indexer,
        HookType.REPLACE,
    )
    _registered = True
