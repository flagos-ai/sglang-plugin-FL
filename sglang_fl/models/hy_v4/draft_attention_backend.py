# SPDX-License-Identifier: Apache-2.0
"""Registration-time HYV4 draft sink routing; no numerical changes.

Native draft factories bypass the model's attention registry. Preserve their
container/metadata behavior while selecting the already existing HYV4 backend
for this model family. Install before model/backend construction and capture.
"""

from functools import wraps


def prepare_patches():
    from sglang.srt.layers.attention.dsa_backend import (
        DeepseekSparseAttnMultiStepBackend,
    )
    from sglang.srt.speculative.draft_utils import DraftBackendFactory

    from .attention_backend import HYV4SparseAttnBackend
    from .register import _hy4

    class HYV4SparseAttnMultiStepBackend(DeepseekSparseAttnMultiStepBackend):
        # Exact native constructor fields/kwargs, changing only child backend
        # class. All metadata/graph methods are inherited without replacement.
        def __init__(
            self,
            model_runner,
            topk,
            speculative_num_steps,
            seed_dsa_topk_from_draft_extend=False,
        ):
            self.topk = topk
            self.speculative_num_steps = speculative_num_steps
            self.attn_backends = []
            for i in range(self.speculative_num_steps - 1):
                self.attn_backends.append(
                    HYV4SparseAttnBackend(
                        model_runner,
                        speculative_step_id=i,
                        topk=self.topk,
                        speculative_num_steps=self.speculative_num_steps,
                        seed_dsa_topk_from_draft_extend=seed_dsa_topk_from_draft_extend,
                    )
                )

    previous_decode = DraftBackendFactory._create_dsa_decode_backend
    previous_extend = DraftBackendFactory._create_dsa_prefill_backend

    @wraps(previous_decode)
    def create_decode(factory):
        if not _hy4(factory.draft_model_runner.model_config.hf_config):
            return previous_decode(factory)
        return (
            "dsa",
            HYV4SparseAttnMultiStepBackend(
                factory.draft_model_runner,
                factory.topk,
                factory.speculative_num_steps,
                seed_dsa_topk_from_draft_extend=factory.seed_dsa_topk_from_draft_extend,
            ),
        )

    @wraps(previous_extend)
    def create_extend(factory):
        if not _hy4(factory.draft_model_runner.model_config.hf_config):
            return previous_extend(factory)
        return (
            "dsa",
            HYV4SparseAttnBackend(
                factory.draft_model_runner,
                skip_prefill=False,
                seed_dsa_topk_from_draft_extend=factory.seed_dsa_topk_from_draft_extend,
            ),
        )

    return [
        (
            DraftBackendFactory,
            "_create_dsa_decode_backend",
            previous_decode,
            create_decode,
        ),
        (
            DraftBackendFactory,
            "_create_dsa_prefill_backend",
            previous_extend,
            create_extend,
        ),
    ]
