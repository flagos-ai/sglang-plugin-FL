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

"""Compressed-tensors target-prefix normalization adapter.

Bare ``ForCausalLM``-style checkpoints (e.g. Qwen-0812) carry compressed-tensors
config targets like ``model.layers.*`` / ``mtp.*``, while the vision-optional
wrapper models (``Qwen3_5MoeForConditionalGeneration`` and co.) build the inner
language model under ``model.language_model.*``. Core v0.5.11's
``CompressedTensorsConfig.from_config`` normalizes those prefixes; this adapter
applies the same normalization by wrapping
``CompressedTensorsConfig._quantization_scheme_map_from_config`` so the plugin
works against unmodified core.

The wrapped method runs before any ``hf_to_sglang_mapper`` transform, so the
normalized keys flow through the exact same pipeline as the core change.
"""
import functools
import logging

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)

logger = logging.getLogger(__name__)

_METHOD = "_quantization_scheme_map_from_config"


def _normalize_targets(target_scheme_map: dict) -> dict:
    """Map bare-model targets to the wrapper's constructed prefix.

    Idempotent: already-prefixed (``model.language_model.*``) targets are left
    untouched.
    """
    return {
        (
            key.replace("model.", "model.language_model.", 1)
            if key.startswith("model.layers.")
            else ("model.language_model." + key if key.startswith("mtp.") else key)
        ): scheme
        for key, scheme in target_scheme_map.items()
    }


def register_compressed_tensors_normalize() -> bool:
    """Wrap the scheme-map builder to normalize bare-model targets.

    Idempotent; returns True when the patch was newly applied.
    """
    desc = CompressedTensorsConfig.__dict__.get(_METHOD)
    if desc is None:
        logger.warning(
            "sglang_fl: %s.%s not found, skip", CompressedTensorsConfig.__name__, _METHOD
        )
        return False
    if getattr(desc.__func__, "_sglang_fl_normalize_patched", False):
        return False
    orig_fn = desc.__func__
    is_classmethod = isinstance(desc, classmethod)

    @functools.wraps(orig_fn)
    def _patched(cls, config, *args, **kwargs):
        scheme_map = orig_fn(cls, config, *args, **kwargs)
        return _normalize_targets(scheme_map)

    _patched._sglang_fl_normalize_patched = True
    # class __dict__ is a mappingproxy — use setattr, not item assignment
    setattr(CompressedTensorsConfig, _METHOD, classmethod(_patched) if is_classmethod else _patched)
    logger.info(
        "sglang_fl: compressed-tensors target normalization patched on %s",
        CompressedTensorsConfig.__name__,
    )
    return True
