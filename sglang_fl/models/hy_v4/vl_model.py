# SPDX-License-Identifier: Apache-2.0
"""Text-only compatibility wrapper for the checkpoint's existing VL class name.

No vision encoder, image/video processor, or media inference is implemented.
"""

from collections.abc import Iterable

import torch
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix
from torch import nn

from .model import HYV4_PACKED_MODULES_MAPPING, HYV4ForCausalLM

_UNMAPPED_TEXT_NORM_NAMES = (".self_attn.query_layernorm.", ".self_attn.key_layernorm.")


def reject_media(forward_batch):
    contains = getattr(forward_batch, "contains_mm_inputs", None)
    if callable(contains) and contains():
        raise ValueError(
            "Hy4-preview draft supports text-only requests; media is unsupported"
        )


class HYV4VLLanguageModel(HYV4ForCausalLM):
    def get_input_embeddings(self):
        return self.model.embed_tokens


class HYV4VLForConditionalGeneration(nn.Module):
    packed_modules_mapping = HYV4_PACKED_MODULES_MAPPING
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_prefix={
            "vit.vit.": "visual.",
            "vit.": "visual.",
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        },
        orig_to_new_substr={"mlp.gate.wg": "mlp.gate"},
    )

    @classmethod
    def shared_experts_fusion_disable_reason(cls, hf_config, quant_config):
        return HYV4ForCausalLM.shared_experts_fusion_disable_reason(
            hf_config.text_config, quant_config
        )

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        if getattr(config, "language_model_only", None) is not True:
            raise ValueError("Hy4-preview wrapper requires text-only configuration")
        self.config = config
        self.pp_group = get_parallel().pp_group
        self.language_model_only = True
        self.language_model = HYV4VLLanguageModel(
            config.text_config,
            quant_config,
            prefix=add_prefix("language_model", prefix),
            use_mrope=False,
        )

    @property
    def start_layer(self):
        return self.language_model.start_layer

    @property
    def end_layer(self):
        return self.language_model.end_layer

    def pad_input_ids(self, input_ids, mm_inputs):
        if mm_inputs is not None and getattr(mm_inputs, "mm_items", None):
            raise ValueError("Hy4-preview draft does not accept image/video items")
        return input_ids

    def prepare_cp_inputs(self, forward_batch, **kwargs):
        raise NotImplementedError(
            "Hy4-preview text-only draft does not qualify context parallelism"
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        pp_proxy_tensors: PPProxyTensors | None = None,
        input_embeds=None,
    ):
        reject_media(forward_batch)
        if input_embeds is None and self.pp_group.is_first_rank:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            buffer = getattr(forward_batch, "input_embeds", None)
            if buffer is not None:
                buffer.copy_(input_embeds)
                input_embeds = buffer
        return self.language_model(
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    def get_embed_and_head(self):
        return self.language_model.get_embed_and_head()

    def post_load_weights(self):
        self.language_model.post_load_weights()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        def language_weights():
            for name, loaded_weight in self.hf_to_sglang_mapper.apply(weights):
                if any(alias in name for alias in _UNMAPPED_TEXT_NORM_NAMES):
                    raise ValueError("Unsupported checkpoint text attention norm alias")
                if name.startswith("visual."):
                    # Checkpoint may bundle unused media weights. No tower is built.
                    continue
                if not name.startswith("language_model."):
                    raise ValueError(
                        f"Unmapped Hy4-preview text checkpoint weight: {name}"
                    )
                yield name.removeprefix("language_model."), loaded_weight

        self.language_model.load_weights(language_weights())


EntryClass = [HYV4VLForConditionalGeneration]
