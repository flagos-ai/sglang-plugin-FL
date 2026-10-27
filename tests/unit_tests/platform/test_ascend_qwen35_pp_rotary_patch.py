# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only regressions for Qwen3.5/3.6 Ascend pipeline RoPE initialization."""

import inspect
import sys
from types import ModuleType, SimpleNamespace

import pytest


def _install_fake_qwen35(monkeypatch, *, bad_init=False, bad_prepare=False):
    sglang = ModuleType("sglang")
    srt = ModuleType("sglang.srt")
    models = ModuleType("sglang.srt.models")
    qwen35 = ModuleType("sglang.srt.models.qwen3_5")

    class Rotary:
        def __init__(self):
            self.position_sin = None
            self.position_cos = None
            self.calls = []

        def get_cos_sin_with_position(self, positions):
            self.calls.append(positions)
            self.position_sin = ("sin", positions)
            self.position_cos = ("cos", positions)

    class Qwen3_5AttentionDecoderLayer:
        def __init__(self, config, layer_id, rotary):
            self.config = config
            self.attn = SimpleNamespace(layer_id=layer_id)
            self.rotary_emb = rotary
            self.prepare_calls = []

        if bad_prepare:

            def forward_prepare_npu(self, renamed):
                return renamed

        else:

            def forward_prepare_npu(self, positions, hidden_states, forward_batch):
                if self.attn.layer_id == self.config.full_attention_interval - 1:
                    self.rotary_emb.get_cos_sin_with_position(positions)
                # Model the native kernel's non-null AND current-position contract.
                assert self.rotary_emb.position_sin == ("sin", positions)
                assert self.rotary_emb.position_cos == ("cos", positions)
                self.prepare_calls.append((positions, hidden_states, forward_batch))
                return hidden_states, "key", "value", "gate"

    class Qwen3_5ForCausalLM:
        if bad_init:

            def __init__(self, renamed):
                self.config = renamed

        else:

            def __init__(self, config, quant_config=None, prefix="", is_nextn=False):
                self.init_args = (config, quant_config, prefix, is_nextn)
                self.config = config
                self.pp_group = SimpleNamespace(world_size=config.pp_size)
                self.start_layer = config.stage_start
                self.end_layer = config.stage_end
                self.rotary = Rotary()
                self.layers = []
                for layer_id in range(config.num_hidden_layers):
                    if (
                        self.start_layer <= layer_id < self.end_layer
                        and layer_id in config.full_attention_layers
                    ):
                        layer = Qwen3_5AttentionDecoderLayer(
                            config, layer_id, self.rotary
                        )
                    else:
                        layer = SimpleNamespace(kind="linear-attention-or-PP-missing")
                    self.layers.append(layer)

    class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM):
        def __init__(self, config, quant_config=None, prefix=""):
            super().__init__(config, quant_config=quant_config, prefix=prefix)

    qwen35.Qwen3_5AttentionDecoderLayer = Qwen3_5AttentionDecoderLayer
    qwen35.Qwen3_5ForCausalLM = Qwen3_5ForCausalLM
    qwen35.Qwen3_5MoeForCausalLM = Qwen3_5MoeForCausalLM
    models.qwen3_5 = qwen35
    srt.models = models
    sglang.srt = srt
    for module in (sglang, srt, models, qwen35):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return qwen35


def _config(start, end, *, pp_size=3):
    return SimpleNamespace(
        full_attention_interval=4,
        full_attention_layers=(3, 7, 11, 15, 19),
        num_hidden_layers=20,
        stage_start=start,
        stage_end=end,
        pp_size=pp_size,
    )


def _local_attention(model, module):
    return [
        layer
        for layer in model.layers[model.start_layer : model.end_layer]
        if isinstance(layer, module.Qwen3_5AttentionDecoderLayer)
    ]


@pytest.mark.parametrize("model_name", ["Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM"])
@pytest.mark.parametrize("start,end", [(0, 6), (6, 16), (16, 20)])
def test_every_uneven_pipeline_stage_refreshes_once_per_forward(
    monkeypatch, model_name, start, end
):
    from sglang_fl.dispatch.backends.vendor.ascend.patches.qwen35_pp_rotary import (
        patch_qwen35_pp_rotary_cache,
    )

    module = _install_fake_qwen35(monkeypatch)
    patch_qwen35_pp_rotary_cache()
    config = _config(start, end)
    extra_args = {"is_nextn": True} if model_name == "Qwen3_5ForCausalLM" else {}
    model = getattr(module, model_name)(
        config, quant_config="quant", prefix="model", **extra_args
    )
    assert model.init_args == (config, "quant", "model", bool(extra_args))
    layers = _local_attention(model, module)
    hidden_states = object()
    # Include equal-sized batches with changed position values and a changed
    # shape, exercising repeated decode as well as graph capture batch sizes.
    for positions, mode in [((0,), "capture"), ((19,), "decode"), ((20, 21), "decode")]:
        forward_batch = SimpleNamespace(mode=mode)
        for layer in layers:
            result = layer.forward_prepare_npu(positions, hidden_states, forward_batch)
            assert result == (hidden_states, "key", "value", "gate")
            assert layer.prepare_calls[-1] == (
                positions,
                hidden_states,
                forward_batch,
            )
    assert model.rotary.calls == [(0,), (19,), (20, 21)]


def test_second_stage_reproduces_null_cache_without_patch(monkeypatch):
    module = _install_fake_qwen35(monkeypatch)
    model = module.Qwen3_5ForCausalLM(_config(6, 16))
    with pytest.raises(AssertionError):
        _local_attention(model, module)[0].forward_prepare_npu((0,), object(), object())
    assert model.rotary.position_sin is None


def test_pp_one_preserves_original_refresh_and_no_local_marker(monkeypatch):
    from sglang_fl.dispatch.backends.vendor.ascend.patches.qwen35_pp_rotary import (
        _LOCAL_FIRST_MARKER,
        patch_qwen35_pp_rotary_cache,
    )

    module = _install_fake_qwen35(monkeypatch)
    patch_qwen35_pp_rotary_cache()
    model = module.Qwen3_5ForCausalLM(_config(0, 20, pp_size=1))
    for layer in _local_attention(model, module):
        assert not getattr(layer, _LOCAL_FIRST_MARKER, False)
        layer.forward_prepare_npu((8,), object(), object())
    assert model.rotary.calls == [(8,)]


def test_stage_without_full_attention_needs_no_cache(monkeypatch):
    from sglang_fl.dispatch.backends.vendor.ascend.patches.qwen35_pp_rotary import (
        patch_qwen35_pp_rotary_cache,
    )

    module = _install_fake_qwen35(monkeypatch)
    patch_qwen35_pp_rotary_cache()
    model = module.Qwen3_5ForCausalLM(_config(4, 7))
    assert _local_attention(model, module) == []
    assert model.rotary.calls == []


def test_interleaved_models_refresh_a_process_shared_rotary_cache(monkeypatch):
    from sglang_fl.dispatch.backends.vendor.ascend.patches.qwen35_pp_rotary import (
        patch_qwen35_pp_rotary_cache,
    )

    module = _install_fake_qwen35(monkeypatch)
    patch_qwen35_pp_rotary_cache()
    models = [
        module.Qwen3_5ForCausalLM(_config(0, 6)),
        module.Qwen3_5ForCausalLM(_config(6, 13)),
        module.Qwen3_5MoeForCausalLM(_config(13, 20)),
        module.Qwen3_5MoeForCausalLM(_config(0, 20, pp_size=1)),
    ]
    # get_rope() caches a shared object in the real process. Interleave dense,
    # MoE, PP=1 and later PP stages instead of assuming each owns its cache.
    shared = models[0].rotary
    for model in models:
        model.rotary = shared
        for layer in _local_attention(model, module):
            layer.rotary_emb = shared
    for step in range(80):
        model = models[step % len(models)]
        positions = (step, step + 1)
        before = len(shared.calls)
        for layer in _local_attention(model, module):
            layer.forward_prepare_npu(positions, object(), object())
        assert len(shared.calls) == before + 1
        assert shared.position_sin == ("sin", positions)
        assert shared.position_cos == ("cos", positions)


def test_patch_is_idempotent_and_preserves_signatures(monkeypatch):
    from sglang_fl.dispatch.backends.vendor.ascend.patches.qwen35_pp_rotary import (
        patch_qwen35_pp_rotary_cache,
    )

    module = _install_fake_qwen35(monkeypatch)
    original_init = module.Qwen3_5ForCausalLM.__init__
    original_prepare = module.Qwen3_5AttentionDecoderLayer.forward_prepare_npu
    patch_qwen35_pp_rotary_cache()
    patched_init = module.Qwen3_5ForCausalLM.__init__
    patched_prepare = module.Qwen3_5AttentionDecoderLayer.forward_prepare_npu
    patch_qwen35_pp_rotary_cache()
    assert module.Qwen3_5ForCausalLM.__init__ is patched_init
    assert module.Qwen3_5AttentionDecoderLayer.forward_prepare_npu is patched_prepare
    assert patched_init.__wrapped__ is original_init
    assert patched_prepare.__wrapped__ is original_prepare
    assert inspect.signature(patched_init) == inspect.signature(original_init)
    assert inspect.signature(patched_prepare) == inspect.signature(original_prepare)


@pytest.mark.parametrize("bad_init,bad_prepare", [(True, False), (False, True)])
def test_incompatible_signature_fails_before_partial_patching(
    monkeypatch, bad_init, bad_prepare
):
    from sglang_fl.dispatch.backends.vendor.ascend.patches.qwen35_pp_rotary import (
        patch_qwen35_pp_rotary_cache,
    )

    module = _install_fake_qwen35(
        monkeypatch, bad_init=bad_init, bad_prepare=bad_prepare
    )
    original_init = module.Qwen3_5ForCausalLM.__init__
    original_prepare = module.Qwen3_5AttentionDecoderLayer.forward_prepare_npu
    with pytest.raises(RuntimeError, match="Unsupported SGLang Qwen3.5 interface"):
        patch_qwen35_pp_rotary_cache()
    assert module.Qwen3_5ForCausalLM.__init__ is original_init
    assert module.Qwen3_5AttentionDecoderLayer.forward_prepare_npu is original_prepare


def test_incomplete_patch_is_rejected_without_double_wrapping(monkeypatch):
    from sglang_fl.dispatch.backends.vendor.ascend.patches.qwen35_pp_rotary import (
        _PATCH_MARKER,
        patch_qwen35_pp_rotary_cache,
    )

    module = _install_fake_qwen35(monkeypatch)
    original_prepare = module.Qwen3_5AttentionDecoderLayer.forward_prepare_npu
    monkeypatch.setattr(
        module.Qwen3_5ForCausalLM.__init__, _PATCH_MARKER, True, raising=False
    )
    with pytest.raises(RuntimeError, match="Incomplete SGLang Qwen3.5 Ascend PP"):
        patch_qwen35_pp_rotary_cache()
    assert module.Qwen3_5AttentionDecoderLayer.forward_prepare_npu is original_prepare
