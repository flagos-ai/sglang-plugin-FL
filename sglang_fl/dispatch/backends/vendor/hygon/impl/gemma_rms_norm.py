# Bridge: GemmaRMSNorm
#
# SGLang signature:
#   forward_cuda(self, x, residual=None, post_residual_addition=None)
#     -> Tensor | tuple[Tensor, Tensor]
#
# Dispatch signature:
#   fn(obj, x, residual=None) -> Tensor | tuple[Tensor, Tensor]
#
# SGLang-specific handling:
#   - post_residual_addition: added to residual before passing to dispatch
#   - GemmaRMSNorm uses weight+1 semantics (handled by reference/vendor impls)

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from sglang.srt.utils import get_bool_env_var
_use_aiter = not get_bool_env_var("SGLANG_PLUGIN_NO_AITER")

def gemma_rms_norm_hcu(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
	w = obj.gemma_weight.data
	if _use_aiter:
		from aiter import rmsnorm2d_fwd as rms_norm
		from aiter import rmsnorm2d_fwd_with_add as fused_add_rms_norm
		if residual is not None:
			output = torch.empty_like(x)
			residual_out = torch.empty_like(x)
			fused_add_rms_norm(
				output, x, residual, residual_out, w, obj.variance_epsilon
			)
			return output, residual_out
		return rms_norm(x, w, obj.variance_epsilon)
	else:
		from vllm._custom_ops import rms_norm
		from lightop import gemma_fused_add_rmsnorm as gemma_fused_add_rmsnorm_dcu
		if not x.is_contiguous():
			x = x.contiguous()
		if residual is not None:
			out, residual_out=gemma_fused_add_rmsnorm_dcu(
                        x, residual, w, obj.variance_epsilon
                    )
			return out, residual_out
		out = torch.empty_like(x)
		rms_norm(out, x, w, obj.variance_epsilon)
		return out
