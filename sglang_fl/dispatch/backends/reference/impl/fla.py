# Reference FLA operator implementations via SGLang's original triton kernels.

from __future__ import annotations


def _original(fn_name: str):
    from sglang_fl.dispatch.fla_patch import get_original

    fn = get_original(fn_name)
    if fn is None:
        raise RuntimeError(
            f"FLA original '{fn_name}' not available — fla_patch not applied yet"
        )
    return fn


def chunk_gated_delta_rule_torch(q, k, v, g, beta, scale=None,
                                 initial_state=None, initial_state_indices=None,
                                 cu_seqlens=None, head_first=False,
                                 use_qk_l2norm_in_kernel=False):
    return _original("chunk_gated_delta_rule")(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens, head_first=head_first,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def fused_recurrent_gated_delta_rule_torch(q, k, v, g, beta=None, scale=None,
                                           initial_state=None,
                                           output_final_state=True,
                                           cu_seqlens=None,
                                           ssm_state_indices=None,
                                           num_accepted_tokens=None,
                                           use_qk_l2norm_in_kernel=False):
    return _original("fused_recurrent_gated_delta_rule")(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def fused_recurrent_gated_delta_rule_packed_decode_torch(
        mixed_qkv, a, b, A_log, dt_bias, scale,
        initial_state, out, ssm_state_indices,
        use_qk_l2norm_in_kernel=False):
    return _original("fused_recurrent_gated_delta_rule_packed_decode")(
        mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log,
        dt_bias=dt_bias, scale=scale,
        initial_state=initial_state, out=out,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
