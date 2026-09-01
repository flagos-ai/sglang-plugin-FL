import torch
import torch.nn as nn
from typing import Optional
from sglang.srt.layers.attention.vision import SingletonCache
from sglang.srt.environ import envs
from sglang.srt.layers.attention.vision import resolve_seqlens
from sglang.srt.layers.dp_attention import get_attention_tp_size


class GCU_VisionFlash3Attention:
    def __init__(
        self,
        **kwargs,
    ):
        torch.nn.Module.__init__(self)
        use_data_parallel = (
            kwargs["use_data_parallel"] if "use_data_parallel" in kwargs else False
        )
        self.tp_size = 1 if use_data_parallel else get_attention_tp_size()

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            cu_seqlens: torch.Tensor | SingletonCache | None,
            bsz: int,
            seq_len: int,
            softmax_scale: Optional[float] = None,
            **kwargs,
        ) -> torch.Tensor:
            from flash_attn.vllm_flash_attn import flash_attn_varlen_func

            r"""
            Args:
                cu_seqlens: [b]
            Returns:
                [b * s, h, head_size]
            """
            window_size = kwargs.get("window_size", (-1, -1))
            s_aux = kwargs.get("s_aux", None)

            if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
                max_seqlen = cu_seqlens[1]
                fa_kwargs = dict(
                    cu_seqlens_q=cu_seqlens[0],
                    cu_seqlens_k=cu_seqlens[0],
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    softmax_scale=softmax_scale,
                    window_size=window_size,
                )
                if s_aux is not None:
                    fa_kwargs["sinks"] = s_aux
                output = flash_attn_varlen_func(q, k, v, **fa_kwargs)
            else:
                cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device=q.device)
                cu_seqlens = cu_seqlens.to(dtype=torch.int32).to(q.device)
                seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
                max_seqlen = seq_lens.max().item()

                fa_kwargs = dict(
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    softmax_scale=softmax_scale,
                    window_size=window_size,
                )
                if s_aux is not None:
                    fa_kwargs["sinks"] = s_aux
                output = flash_attn_varlen_func(q, k, v, **fa_kwargs)

            return output

def patch_vision_flashattention_backend():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    VISION_FLASH3_ATTENTION = "sglang.srt.layers.attention.vision.VisionFlash3Attention"
    HookRegistry.register(f"{VISION_FLASH3_ATTENTION}.__init__", GCU_VisionFlash3Attention.__init__, HookType.REPLACE)
    HookRegistry.register(f"{VISION_FLASH3_ATTENTION}.forward", GCU_VisionFlash3Attention.forward, HookType.REPLACE)
