# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""V4.1 PD admission follows attention geometry and transfer strides."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "isolated_dsv41_pd",
    Path(__file__).parents[3] / "sglang_fl/dsv41_runtime/pd.py",
)
pd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pd)


@pytest.mark.parametrize("dp", [4, 8])
def test_decode_attention_layouts(dp):
    cfg = SimpleNamespace(
        tp_size=8,
        ep_size=8,
        dp_size=dp,
        enable_dp_attention=True,
        enable_dp_lm_head=True,
        disaggregation_mode="decode",
        moe_a2a_backend="none",
    )
    assert pd.allow_decode_config(cfg)
    for key, bad in (
        ("disaggregation_mode", "prefill"),
        ("enable_dp_attention", False),
        ("enable_dp_lm_head", False),
        ("moe_a2a_backend", "deepep"),
        ("ep_size", 4),
        ("tp_size", 16),
        ("dp_size", 2),
    ):
        changed = SimpleNamespace(**vars(cfg))
        setattr(changed, key, bad)
        assert not pd.allow_decode_config(changed)


@pytest.mark.parametrize("tp,dp", [(1, 8), (2, 4)])
def test_peer_requires_matching_full_tp_ep(tp, dp):
    args = dict(
        is_mla=True,
        local_tp=8,
        local_ep=8,
        local_attn_tp=tp,
        local_attn_dp=dp,
        remote_attn_tp=8,
        remote_dp=1,
    )
    assert pd.allow_peer_topology(**args)
    for key, bad in (
        ("is_mla", False),
        ("local_ep", 4),
        ("remote_dp", 4),
        ("remote_attn_tp", 4),
    ):
        assert not pd.allow_peer_topology(**{**args, key: bad})


def test_transfer_strides_checked_before_write():
    pd.validate_c2_strides([384, 1024], (384, 1024))
    for destination in ([384, 1025], [384], [1024, 384]):
        with pytest.raises(ValueError, match="strides differ"):
            pd.validate_c2_strides([384, 1024], destination)


@pytest.mark.parametrize("model_api", ["method", "native_resolver"])
def test_flagcx_validation_preserves_native_feature_checks(monkeypatch, model_api):
    spec = importlib.util.spec_from_file_location(
        "isolated_dsv41_model.patch",
        Path(__file__).parents[3] / "sglang_fl/dsv41_runtime/patch.py",
    )
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    import sys
    import types

    package = types.ModuleType("isolated_dsv41_model")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, package.__name__ + ".pd", pd)
    cfg = SimpleNamespace(
        tp_size=8,
        ep_size=8,
        dp_size=1,
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        disaggregation_mode="prefill",
        disaggregation_transfer_backend="flagcx",
        speculative_algorithm="DSPARK",
        attn_cp_size=1,
    )
    args = SimpleNamespace(
        config=cfg,
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v41")
        ),
    )
    scope = {"resolving_view": lambda x: x.config}
    if model_api == "native_resolver":
        model = args.get_model_config()
        del args.get_model_config
        scope["model_config_of"] = lambda x: model
    exec(
        "def validate(args):\n"
        "    cfg = resolving_view(args)\n"
        "    assert cfg.disaggregation_transfer_backend == 'mooncake'\n"
        "    if cfg.attn_cp_size != 1: raise ValueError('native CP check')\n"
        "    return cfg.tp_size\n",
        scope,
    )
    assert policy._validate_features(scope["validate"], args) == 8
    assert cfg.disaggregation_transfer_backend == "flagcx"
    cfg.attn_cp_size = 2
    with pytest.raises(ValueError, match="native CP check"):
        policy._validate_features(scope["validate"], args)
    cfg.attn_cp_size = 1
    cfg.dp_size = 8
    cfg.enable_dp_attention = cfg.enable_dp_lm_head = True
    cfg.disaggregation_mode = "decode"
    cfg.moe_a2a_backend = "none"
    # DP8 can be a native Mooncake layout, but is not admitted for FlagCX.
    with pytest.raises(ValueError, match="TP2/DP4"):
        policy._validate_features(scope["validate"], args)
    cfg.dp_size = 4
    assert policy._validate_features(scope["validate"], args) == 8
