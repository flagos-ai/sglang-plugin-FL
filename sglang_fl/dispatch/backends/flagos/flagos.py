# FlagOS backend class.

from __future__ import annotations

from .. import Backend


class FlagOSBackend(Backend):
    """FlagOS default backend — Triton-based implementations from FlagGems and other FlagOS libraries."""

    _available = None

    @property
    def name(self) -> str:
        return "flagos"

    def is_available(self) -> bool:
        if FlagOSBackend._available is None:
            try:
                import flag_gems  # noqa: F401

                FlagOSBackend._available = True
            except ImportError:
                FlagOSBackend._available = False
        return FlagOSBackend._available

    def silu_and_mul(self, obj, x):
        from .impl.activation import silu_and_mul_flagos

        return silu_and_mul_flagos(obj, x)

    def rms_norm(self, obj, x, residual=None):
        from .impl.normalization import rms_norm_flagos

        return rms_norm_flagos(obj, x, residual)

    def rotary_embedding(
        self,
        obj,
        query,
        key,
        cos,
        sin,
        position_ids,
        rotary_interleaved=False,
        inplace=True,
    ):
        from .impl.rotary import rotary_embedding_flagos

        return rotary_embedding_flagos(
            obj, query, key, cos, sin, position_ids, rotary_interleaved, inplace
        )

    def topk(
        self,
        obj,
        hidden_states,
        router_logits,
        *,
        num_token_non_padded=None,
        expert_location_dispatch_info=None,
    ):
        from .impl.topk import topk_flagos

        return topk_flagos(
            obj,
            hidden_states,
            router_logits,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )

    def fused_recurrent_gated_delta_rule(
        self,
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=None,
        ssm_state_indices=None,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=False,
    ):
        from .impl.fla import fused_recurrent_gated_delta_rule_flagos

        return fused_recurrent_gated_delta_rule_flagos(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            output_final_state,
            cu_seqlens,
            ssm_state_indices,
            num_accepted_tokens,
            use_qk_l2norm_in_kernel,
        )
