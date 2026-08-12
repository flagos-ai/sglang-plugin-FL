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

"""Qwen3.5 MoE bare-text (no vision) coverage adapter.

SGLang core models ``Qwen3_5MoeForConditionalGeneration``, a vision-optional
MoE wrapper, but its config registry only knows the *wrapped* ``qwen3_5_moe``
model_type. Some checkpoints (e.g. Qwen-0812, 2.3T) ship as **bare text**
configs: ``model_type: qwen3_5_moe_text``, architecture
``Qwen3_5MoeForCausalLM``, no ``vision_config``, and ``model.layers.*``
weights directly (instead of ``model.language_model.*``).

This adapter teaches sglang_fl about the bare variant with zero core changes:

1. Register ``Qwen3_5MoeTextConfig`` into sglang's ``_CONFIG_REGISTRY`` so
   ``get_config()`` re-resolves the ``qwen3_5_moe_text`` model_type to the
   sglang config class — which carries ``layers_block_type`` and
   ``norm_topk_prob=True`` needed by ``models/qwen2_moe.py`` TopK. Re-resolving
   also means core ``qwen3_5.py`` never sees a native transformers config, so
   no ``layer_types`` fallback is needed in core.
2. Register a bare ``Qwen3_5MoeForCausalLM`` wrapper into the model registry
   so the model loader routes the architecture to the vision-optional MoE
   wrapper, reusing its forward / load_weights / lm_head / MTP hooks.
3. Patch the core wrapper ``__init__``s to tolerate ``self.visual is None``
   (bare text configs have no vision tower); core v0.5.11 unconditionally
   reads ``self.visual.deepstack_visual_indexes`` after init.

Registration happens inside ``load_plugin()``, which sglang runs before any
model is loaded, so everything takes effect for every server launch.
"""
import functools
import logging

from sglang.srt.configs.qwen3_5 import Qwen3_5MoeTextConfig
from sglang.srt.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.utils.hf_transformers.common import _CONFIG_REGISTRY

logger = logging.getLogger(__name__)

_MODEL_TYPE = "qwen3_5_moe_text"
_ARCH = "Qwen3_5MoeForCausalLM"

# Core classes whose __init__ unconditionally reads self.visual when building
# the (vision-optional) wrapper; harmless to patch for configs that do have a
# visual tower, and required for bare text configs where self.visual is None.
_WRAPPER_CLASSES = (
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
)


class Qwen3_5MoeForCausalLM(Qwen3_5MoeForConditionalGeneration):
    """Bare Qwen3.5 MoE arch (no vision), e.g. Qwen-0812.

    The vision-optional MoE wrapper builds its inner language model with the
    core ``language_model_cls`` (the bare-LM class in
    ``sglang.srt.models.qwen3_5``), so this wrapper body can stay empty.
    """


def _patch_core_visual_guard() -> None:
    """Make core wrapper __init__ tolerate configs without a vision tower.

    Core v0.5.11 does::

        self.deepstack_visual_indexes = self.visual.deepstack_visual_indexes

    right after super().__init__, which raises AttributeError when the config
    has no ``vision_config`` (bare text checkpoints). Wrap each __init__ so an
    AttributeError raised purely because ``self.visual is None`` is recovered
    by setting ``deepstack_visual_indexes = None`` (and the MoE-only
    ``num_fused_shared_experts = 0`` that core sets right after). Any other
    exception is re-raised untouched.
    """
    import importlib

    m = importlib.import_module("sglang.srt.models.qwen3_5")
    for name in _WRAPPER_CLASSES:
        cls = getattr(m, name, None)
        if cls is None:
            continue
        if getattr(cls, "_sglang_fl_visual_guard_patched", False):
            continue
        orig_init = cls.__init__

        @functools.wraps(orig_init)
        def _patched_init(self, *args, **kwargs):
            try:
                orig_init(self, *args, **kwargs)
            except AttributeError:
                if self.visual is None:
                    # bare text config: no vision tower
                    self.deepstack_visual_indexes = None
                    if not hasattr(self, "num_fused_shared_experts"):
                        self.num_fused_shared_experts = 0
                else:
                    raise

        cls.__init__ = _patched_init
        cls._sglang_fl_visual_guard_patched = True
        logger.info("sglang_fl: visual guard patched on %s", name)


def register_qwen3_5_moe_text() -> bool:
    """Idempotently register bare-text Qwen3.5 MoE config + arch.

    Safe to call more than once; returns True if any registration was newly
    applied. No-ops when the core already covers the model_type / arch.
    """
    applied = False
    if _MODEL_TYPE not in _CONFIG_REGISTRY:
        _CONFIG_REGISTRY[_MODEL_TYPE] = Qwen3_5MoeTextConfig
        logger.info(
            "sglang_fl: registered config %s for model_type %r",
            Qwen3_5MoeTextConfig.__name__,
            _MODEL_TYPE,
        )
        applied = True
    if _ARCH not in ModelRegistry.models:
        ModelRegistry.models[_ARCH] = Qwen3_5MoeForCausalLM
        logger.info("sglang_fl: registered arch %s", _ARCH)
        applied = True
    _patch_core_visual_guard()
    return applied
