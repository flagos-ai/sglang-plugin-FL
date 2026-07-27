import torch
from typing import Optional

def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = -1,
    **kwargs,
):
    if isinstance(activation, bool) and activation:
        activation = "silu"

    original_x_dtype = x.dtype
    x = x.to(conv_states.dtype)
    out = torch.empty_like(x)

    silu_activation = activation in ("silu", "swish")

    if has_initial_state is not None and has_initial_state.dtype != torch.int8:
        has_initial_state = has_initial_state.to(torch.int8)
    weight_prepared = weight.t().contiguous()

    torch.ops.sgl_kernel.causal_conv1d_fwd(
        out,
        conv_states,
        x,
        weight_prepared,
        bias,
        cache_indices,
        has_initial_state,
        query_start_loc,
        silu_activation,
        pad_slot_id,
    )

    return out.to(original_x_dtype)

# ---------------------------------------------------------------------------
# Patch registration
# ---------------------------------------------------------------------------

def patch_causal_conv1d_fn():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register(f"sglang.srt.layers.attention.mamba.causal_conv1d_triton.causal_conv1d_fn", causal_conv1d_fn, HookType.REPLACE)