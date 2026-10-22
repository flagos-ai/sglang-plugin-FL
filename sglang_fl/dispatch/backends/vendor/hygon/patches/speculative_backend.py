"""Register HCU draft backends without changing another vendor's dispatch."""

from __future__ import annotations

from functools import wraps


def _create_hcu_draft_backend(factory, *, decode):
    # Keep kernel imports lazy: this patch also runs before CLI parsing.
    from ..impl.attention_backend import HCUAttnBackend, HCUMultiStepDraftBackend

    if factory.topk != 1:
        raise ValueError(
            "HCU MTP requires speculative_eagle_topk=1; tree attention is unsupported"
        )
    if decode:
        backend = HCUMultiStepDraftBackend(
            factory.draft_model_runner, factory.topk, factory.speculative_num_steps
        )
    else:
        backend = HCUAttnBackend(factory.draft_model_runner, skip_prefill=False)
    return "hcu", backend


def register_hcu_speculative_backends():
    """Extend SGLang's draft factory while retaining its wrappers and stamps.

    SGLang 0.5.18 does not expose a draft backend registration API. Intercept
    only the common factory map, so the upstream steps<=1 shortcut, hybrid
    wrapper and per-child backend stamps retain their original semantics.
    """
    from sglang.srt.speculative.draft_utils import DraftBackendFactory

    original = DraftBackendFactory._create_backend
    if getattr(original, "_hcu_speculative_registered", False):
        return

    @wraps(original)
    def create_backend(
        self, backend_name, backend_map, error_template, stamps_children=False
    ):
        # Each caller owns a fresh map. Never mutate upstream/global mappings.
        backend_map = dict(backend_map)
        backend_map["hcu"] = lambda: _create_hcu_draft_backend(
            self, decode=stamps_children
        )
        return original(
            self,
            backend_name,
            backend_map,
            error_template,
            stamps_children=stamps_children,
        )

    create_backend._hcu_speculative_registered = True
    DraftBackendFactory._create_backend = create_backend
