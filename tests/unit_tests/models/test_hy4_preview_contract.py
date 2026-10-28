# SPDX-License-Identifier: Apache-2.0
"""CPU contracts; these tests do not initialize SGLang or a CUDA device."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODEL = Path(__file__).resolve().parents[3] / "sglang_fl/models/hy_v4"


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"hy4_contract_{name}", MODEL / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "role,expected",
    [
        (False, "target"),
        (True, "draft"),
        (None, "unknown"),
        (0, "unknown"),
        (1, "unknown"),
    ],
)
def test_role_is_per_runner_and_strict_bool(role, expected):
    policy = load("attention_policy")
    runner = SimpleNamespace(is_draft_worker=role)
    assert policy.freeze_role_policy(runner) == expected


@pytest.mark.parametrize(
    "mode,accepted",
    [
        ("DECODE", True),
        ("TARGET_VERIFY", True),
        ("EXTEND", False),
        ("DRAFT_EXTEND_V2", False),
    ],
)
def test_prefill_and_draft_extend_remain_native(mode, accepted):
    policy = load("attention_policy")
    backend = SimpleNamespace(
        _hy4_mla_role="target",
        _hy4_forward_mode=SimpleNamespace(name=mode),
    )
    assert policy.eligible_forward(backend) is accepted
    backend._hy4_mla_role = "unknown"
    assert not policy.eligible_forward(backend)


def test_nextn_keeps_target_checkpoint_depth():
    mod = load("nextn_utils")
    text = SimpleNamespace(num_nextn_predict_layers=1, num_hidden_layers=78)
    config = SimpleNamespace(
        model_type="hy_v4_vl", architectures=["HYV4VLForConditionalGeneration"]
    )
    runner = SimpleNamespace(
        is_draft_model=True,
        hf_config=config,
        hf_text_config=text,
        hidden_size=6144,
        num_nextn_predict_layers=1,
    )
    mod.configure_hy4_draft(runner)
    mod.configure_hy4_hidden_shapes(runner)
    assert config.architectures == ["HYV4ForCausalLMNextN"]
    assert text.num_hidden_layers == 78
    assert runner.num_hidden_layers == runner.num_attention_layers == 1
    assert runner.spec_hidden_size == 6144
    assert runner.hc_hidden_size is None


@pytest.mark.parametrize("layers", [0, 2])
def test_nextn_rejects_missing_or_multiple_mtp(layers):
    mod = load("nextn_utils")
    runner = SimpleNamespace(
        is_draft_model=True,
        hf_config=SimpleNamespace(
            model_type="hy_v4", architectures=["HYV4ForCausalLM"]
        ),
        hf_text_config=SimpleNamespace(num_nextn_predict_layers=layers),
    )
    with pytest.raises(ValueError):
        mod.configure_hy4_draft(runner)


class QuantConfig:
    target_scheme_map = None
    sparsity_scheme_map = None
    sparsity_ignore_list = None

    def get_name(self):
        return "compressed_tensors"


def test_ct_aliases_copy_without_mutating_target():
    mod = load("nextn_utils")
    original = QuantConfig()
    original.target_scheme_map = {
        "model.mtp_layers.0.eh_proj": {"weight": "int8"},
        "unrelated": {"weight": "bf16"},
    }
    original.ignore = ["model.mtp_layers.0.enorm"]
    draft = mod.remap_hy4_mtp_quant_config(original)
    assert draft.target_scheme_map["model.eh_proj"] == {"weight": "int8"}
    assert "model.enorm" in draft.ignore
    assert "model.eh_proj" not in original.target_scheme_map
    assert original.ignore == ["model.mtp_layers.0.enorm"]
    draft.target_scheme_map["model.eh_proj"]["weight"] = "changed"
    assert original.target_scheme_map["model.mtp_layers.0.eh_proj"]["weight"] == "int8"


def test_only_bundled_mtp_weights_are_forwarded():
    mod = load("nextn_utils")
    a, b, unrelated = object(), object(), object()
    values = list(
        mod.iter_hy4_nextn_weights(
            [
                ("model.mtp_layers.0.eh_proj.weight", a),
                ("model.mtp_layers.0.final_layernorm.weight", b),
                ("model.layers.0.weight", unrelated),
                ("model.mtp_layers.1.weight", unrelated),
            ],
            78,
        )
    )
    assert values == [
        ("model.layers.78.eh_proj.weight", a),
        ("model.layers.78.shared_head.norm.weight", b),
    ]


def test_hpc_absence_uses_explicit_model_fallback(monkeypatch):
    mod = load("ihc_kernels")
    monkeypatch.setattr(mod, "_hpc_ihc_op", lambda *args: None)
    with pytest.raises(NotImplementedError):
        mod.fused_hy4_ihc_post(
            torch.zeros(1, 4), torch.zeros(1, 4, 4), torch.zeros(1, 4)
        )


def test_checkpoint_wrapper_config_is_explicitly_text_only():
    mod = load("config")
    config = mod.HYV4VLConfig(
        text_config={"architectures": ["HYV4ForCausalLM"]},
        vision_config={"unused_metadata": 7},
    )
    assert config.architectures == ["HYV4VLForConditionalGeneration"]
    assert config.language_model_only is True
    assert config.text_config.architectures == ["HYV4ForCausalLM"]
    assert config.vision_config == {"unused_metadata": 7}
    assert isinstance(config.vision_config, dict)


def test_checkpoint_wrapper_rejects_media_enable():
    mod = load("config")
    with pytest.raises(ValueError, match="text-only"):
        mod.HYV4VLConfig(language_model_only=False)


def test_hpc_launch_failure_propagates_without_retry(monkeypatch):
    mod = load("ihc_kernels")
    calls = []

    def launch(*args):
        calls.append(args)
        raise RuntimeError("launch failed after selection")

    monkeypatch.setattr(mod, "_hpc_ihc_op", lambda *args: launch)
    output, residual, post = torch.zeros(1, 4), torch.zeros(1, 4, 4), torch.zeros(1, 4)
    with pytest.raises(RuntimeError, match="launch failed after selection"):
        mod.fused_hy4_ihc_post(output, residual, post)
    assert len(calls) == 1
    assert all(a is b for a, b in zip(calls[0], (output, residual, post)))


def test_sparse_mla_dispatch_preserves_native_default_and_vendor_filters(monkeypatch):
    from sglang_fl.dispatch import (
        OpManager,
        SelectionPolicy,
        builtin_ops,
        policy_context,
    )
    from sglang_fl.models.hy_v4 import sparse_mla_active_split as adapter

    calls = []
    monkeypatch.setattr(adapter, "_native_route", lambda *args: calls.append("native"))
    monkeypatch.setattr(
        adapter, "_library_route", lambda *args: calls.append("library")
    )
    # Isolate operator discovery; exercise the repository's actual manager and
    # policy selection, without importing accelerator implementation packages.
    monkeypatch.setattr(builtin_ops, "register_builtins", adapter.register_ops)
    manager = OpManager()
    name = "sparse_mla_active_split"
    with policy_context(SelectionPolicy()):
        manager.resolve(name)()
    with policy_context(SelectionPolicy(prefer="vendor")):
        manager.resolve(name)()
    with policy_context(
        SelectionPolicy(prefer="vendor", deny_vendors=frozenset({"nvidia"}))
    ):
        manager.resolve(name)()
    with policy_context(
        SelectionPolicy.from_dict(per_op_order={name: ["reference", "vendor"]})
    ):
        manager.resolve(name)()
    assert calls == ["native", "library", "native", "native"]


def test_sparse_mla_library_metadata_guard_and_failure_do_not_retry(monkeypatch):
    from sglang_fl.models.hy_v4 import sparse_mla_active_split as adapter

    calls = []
    eligible = True

    def launch(*args, **kwargs):
        calls.append((args, kwargs))
        return "computed", "unused"

    def metadata(*args):
        return object() if eligible else None

    monkeypatch.setattr(adapter, "_public_provider", lambda: (launch, metadata))
    args = tuple(object() for _ in range(6)) + (4,)
    assert adapter._library_route(*args) == "computed"
    assert len(calls) == 1
    assert calls[0][0] == args[:4]
    assert calls[0][1] == {"active_heads": 4, "sink": args[4], "lengths": args[5]}
    eligible = False
    assert adapter._library_route(*args) is None
    assert len(calls) == 1
    eligible = True

    def failure(*args, **kwargs):
        calls.append("failed")
        raise RuntimeError("selected launch failed")

    monkeypatch.setattr(adapter, "_public_provider", lambda: (failure, metadata))
    with pytest.raises(RuntimeError, match="selected launch failed"):
        adapter._library_route(*args)
    assert calls[-1] == "failed"
    assert len(calls) == 2


def test_draft_causal_lengths_and_cache_rows_preserve_native_inputs():
    mod = load("draft_causal_lengths")
    seen = []

    class Child:
        def get_cuda_graph_seq_len_fill_value(self):
            return 0

        def init_forward_metadata(self, view):
            seen.append(view)

    fixed = mod.causal_backend_type(object, None, type("UnusedForwardBatch", (), {}))
    container = fixed(
        SimpleNamespace(attn_backends=[Child(), Child()]), SimpleNamespace(device="cpu")
    )
    batch = SimpleNamespace(
        batch_size=2,
        seq_lens=torch.tensor([5, 10]),
        seq_lens_cpu=torch.tensor([5, 10]),
        seq_lens_sum=15,
        unchanged=object(),
    )
    container.init_forward_metadata(batch)
    assert [view.seq_lens.tolist() for view in seen] == [[6, 11], [7, 12]]
    assert [view.seq_lens_cpu.tolist() for view in seen] == [[6, 11], [7, 12]]
    assert [view.seq_lens_sum for view in seen] == [17, 19]
    assert all(view.unchanged is batch.unchanged for view in seen)
    assert batch.seq_lens.tolist() == batch.seq_lens_cpu.tolist() == [5, 10]
    assert batch.seq_lens_sum == 15
    locations = torch.arange(6, dtype=torch.int64).reshape(2, 3).T
    staged = container.stage_per_step_cache_locs(locations)
    assert torch.equal(staged, locations)
    assert staged.stride(1) == 1
    assert locations.stride(1) == 3
    container._hy4_storage_pinned = True
    with pytest.raises(RuntimeError, match="cannot resize captured storage"):
        container._reserve(3)


def test_registration_rolls_back_related_wrappers_and_preserves_existing_owners(
    monkeypatch,
):
    import sys
    from types import ModuleType

    registration = load("register")
    registration.__package__ = "hy4_registration_fixture"
    registration.__spec__ = importlib.util.spec_from_file_location(
        "hy4_registration_fixture.register", MODEL / "register.py"
    )
    native_draft = lambda self: "draft"
    native_shapes = lambda self: "shapes"
    native_dsa = lambda config: False
    native_skips = lambda config, layer: False
    native_backend = lambda runner: "backend"
    model_config = type(
        "ModelConfig",
        (),
        {
            "_config_draft_model": native_draft,
            "_derive_model_shapes": native_shapes,
        },
    )
    mc = ModuleType("sglang.srt.configs.model_config")
    mc.ModelConfig = model_config
    mc.is_deepseek_dsa, mc.dsa_layer_skips_topk = native_dsa, native_skips
    aliases = ModuleType("sglang.test_hy4_aliases")
    aliases.is_dsa = native_dsa
    configs, models = {"unrelated": object()}, {"unrelated": object()}
    saved_configs, saved_models = configs.copy(), models.copy()
    backends = {"dsa": native_backend}
    mapping = SimpleNamespace(_extra_content=configs)

    class AutoConfig:
        @staticmethod
        def register(key, cls, **kwargs):
            configs[key] = cls

    class Owner:
        fail = True

        def __setattr__(self, name, value):
            if name == "last" and value == "replacement" and self.fail:
                raise RuntimeError("publish failure")
            object.__setattr__(self, name, value)

    owner = Owner()
    owner.decode, owner.prefill, owner.last = "decode", "prefill", "last"
    sink = lambda: [
        (owner, "decode", "decode", "sink"),
        (owner, "prefill", "prefill", "prefill replacement"),
    ]
    causal = lambda previous: [
        (owner, "decode", previous, "causal"),
        (owner, "last", "last", "replacement"),
    ]
    fixtures = {
        "sglang.srt.configs": {"model_config": mc},
        "sglang.srt.configs.model_config": vars(mc),
        "sglang.test_hy4_aliases": vars(aliases),
        "sglang.srt.layers.attention.attention_registry": {
            "ATTENTION_BACKENDS": backends
        },
        "sglang.srt.models.registry": {"ModelRegistry": SimpleNamespace(models=models)},
        "transformers": {"AutoConfig": AutoConfig},
        "transformers.models.auto.configuration_auto": {"CONFIG_MAPPING": mapping},
        "hy4_registration_fixture.config": {
            "HYV4Config": object(),
            "HYV4VLConfig": object(),
        },
        "hy4_registration_fixture.model": {"HYV4ForCausalLM": object()},
        "hy4_registration_fixture.nextn": {"HYV4ForCausalLMNextN": object()},
        "hy4_registration_fixture.vl_model": {
            "HYV4VLForConditionalGeneration": object()
        },
        "hy4_registration_fixture.nextn_utils": {
            "configure_hy4_draft": lambda self: None,
            "configure_hy4_hidden_shapes": lambda self: None,
        },
        "hy4_registration_fixture.draft_attention_backend": {"prepare_patches": sink},
        "hy4_registration_fixture.draft_causal_lengths": {"prepare_patches": causal},
    }
    for name, attrs in fixtures.items():
        module = ModuleType(name)
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(RuntimeError, match="publish failure"):
        registration.register_hy_v4()
    assert not registration._registered
    assert configs == saved_configs and models == saved_models
    assert backends["dsa"] is native_backend
    assert model_config._config_draft_model is native_draft
    assert model_config._derive_model_shapes is native_shapes
    assert sys.modules["sglang.test_hy4_aliases"].is_dsa is native_dsa
    assert (owner.decode, owner.prefill, owner.last) == ("decode", "prefill", "last")
    owner.fail = False
    registration.register_hy_v4()
    assert registration._registered
    assert (owner.decode, owner.prefill, owner.last) == (
        "causal",
        "prefill replacement",
        "replacement",
    )
    installed = model_config._config_draft_model
    registration.register_hy_v4()
    assert model_config._config_draft_model is installed
