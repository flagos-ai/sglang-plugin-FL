from functools import lru_cache
from typing import Optional, Union

import torch
from sgl_kernel.debug_utils import maybe_wrap_debug_kernel

from sglang.srt.environ import envs
from sglang.srt.utils import is_ppu

if not is_ppu():
    try:
        from sgl_kernel import flash_ops
    except:
        raise ImportError(
            "Can not import FA3 in sgl_kernel. Please check your installation."
        )

else:
    try:
        from flash_attn_interface import flash_attn_varlen_func as fa_varlen_func
        from flash_attn_interface import flash_attn_with_kvcache as fa_with_kvcache
        from flash_attn_interface import get_scheduler_metadata as get_metadata
    except:
        raise ImportError(
            "Can not import flash-attention. Please check your installation."
        )
# Add for nvtx profiling
SGLANG_PROFILE_NVTX = envs.SGLANG_PROFILE_NVTX.get()
SGLANG_PROFILE_NVTX_PRINT_SEQLEN = envs.SGLANG_PROFILE_NVTX_PRINT_SEQLEN.get()
if SGLANG_PROFILE_NVTX:
    try:
        from torch.cuda.nvtx import range_pop as th_nvtx_range_pop
        from torch.cuda.nvtx import range_push as th_nvtx_range_push
    except ImportError as e:
        SGLANG_PROFILE_NVTX = False
        SGLANG_PROFILE_NVTX_PRINT_SEQLEN = False


@lru_cache(maxsize=1)
def is_fa3_supported(device=None) -> bool:
    #  There some fa3 FYI
    #  FA3 can fail without a enough shared memory for a some shapes, such as higher
    #  hidden_dim or some special cases.
    #  Right now, fa3 is supported for sm80/sm87 and sm86/sm89. The main different
    #  Between sm80/sm87 and sm86/sm89 is the shared memory size. you can follow the link below for more information
    #  https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-8-x
    #  And for sgl-kernel right now, we can build fa3 on sm80/sm86/sm89/sm90a.
    #  That means if you use A100/A*0/L20/L40/L40s/4090 you can use fa3.
    return (torch.version.cuda >= "12.3") and (
        torch.cuda.get_device_capability(device)[0] == 9
        or torch.cuda.get_device_capability(device)[0] == 8
    )


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


@maybe_wrap_debug_kernel
def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    attention_chunk: Optional[int] = None,
    softcap=0.0,  # 0.0 means deactivated
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,  # Can be tuned for speed
    pack_gqa=None,  # Can be tuned for speed
    sm_margin=0,  # Can be tuned if some SMs are used for communication
    return_softmax_lse=False,
    sinks=None,
    score_mod=None,
    aux_tensors=None,
    ver=3,
    out=None,
):
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

    Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
    rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
    and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
    indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

    See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Note: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no page_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a page_table (i.e. paged KV cache)
            page_block_size must be a multiple of 256.
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim_v) if there's no page_table,
            or (num_blocks, page_block_size, nheads_k, headdim_v) if there's a page_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim_v). Similar to k.
        qv [optional]: (batch_size, seqlen, nheads, headdim_v)
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
        cache_leftpad: (batch_size,), dtype torch.int32. The index that the KV cache starts. If None, assume 0.
        page_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        attention_chunk: Optional[int]. If not None, splits the query into chunks of this size to save memory.
        softcap: float. Anything > 0 activates softcapping attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Don't change this unless you know what you are doing.
        return_softmax_lse: bool. Whether to return the logsumexp of the attention scores.
        score_mod [optional]: A callable that takes the attention scores and applies a modification.
        aux_tensors [optional]: Some score_mods will want to read from global aux_tensors. This is how we thread them through to the inner kernel.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """

    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    if softmax_scale is None:
        softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (
            -0.5
        )
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)

    q, k_cache, k, v = [maybe_contiguous(x) for x in (q, k_cache, k, v)]
    v_cache = (
        v_cache.contiguous()
        if v_cache.stride(-1) != 1 and v_cache.stride(-3) != 1
        else v_cache
    )
    cu_seqlens_q, cu_seqlens_k_new = [
        maybe_contiguous(x) for x in (cu_seqlens_q, cu_seqlens_k_new)
    ]
    page_table, cache_batch_idx, cache_leftpad = [
        maybe_contiguous(x) for x in (page_table, cache_batch_idx, cache_leftpad)
    ]
    rotary_cos, rotary_sin = [maybe_contiguous(x) for x in (rotary_cos, rotary_sin)]
    rotary_seqlens = maybe_contiguous(rotary_seqlens)
    attention_chunk = 0 if attention_chunk is None else int(attention_chunk)

    if SGLANG_PROFILE_NVTX:
        if cu_seqlens_q is not None:
            batch_size = len(cu_seqlens_q) - 1
        else:
            batch_size = 0
        if page_table is not None:
            num_blocks = k_cache.shape[-4]
            page_block_size = k_cache.shape[-3]
            page_table_flag = 1
        else:
            num_blocks = 0
            page_block_size = 0
            page_table_flag = 0

        if torch.cuda.is_current_stream_capturing():
            nvtx_message = f"[FW_FMHA] --format=flash_attn_{ver},Forward,type:D,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_value:{k_cache.shape[-1]},num_heads_k:{k_cache.shape[-2]},num_heads:{q.shape[-2]},batch_size:{batch_size},data_type:{q.dtype},window_size_left:{window_size[0]},window_size_right:{window_size[1]},softcap:{softcap},paged_kv:{page_table_flag}"
        else:
            if SGLANG_PROFILE_NVTX_PRINT_SEQLEN:
                cu_seqlens_q_list = (
                    cu_seqlens_q.flatten().cpu().tolist()
                    if cu_seqlens_q is not None
                    else "[]"
                )
                nvtx_message = f"[FW_FMHA] --format=flash_attn_{ver},Forward,type:P,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_value:{k_cache.shape[-1]},num_heads_k:{k_cache.shape[-2]},num_heads:{q.shape[-2]},batch_size:{batch_size},data_type:{q.dtype},window_size_left:{window_size[0]},window_size_right:{window_size[1]},softcap:{softcap},paged_kv:{page_table_flag},num_blocks:{num_blocks},page_block_size:{page_block_size},cu_seqlens_q:{cu_seqlens_q_list}"
            else:
                nvtx_message = f"[FW_FMHA] --format=flash_attn_{ver},Forward,type:P,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_value:{k_cache.shape[-1]},num_heads_k:{k_cache.shape[-2]},num_heads:{q.shape[-2]},batch_size:{batch_size},data_type:{q.dtype},window_size_left:{window_size[0]},window_size_right:{window_size[1]},softcap:{softcap},paged_kv:{page_table_flag}"
        th_nvtx_range_push(nvtx_message)

    out, softmax_lse, *rest = torch.ops.sgl_kernel.fwd.default(
        q,
        k_cache,
        v_cache,
        k,
        v,
        qv,
        out,  # out (pre-allocated output to avoid DtoD copy)
        cu_seqlens_q,
        None,  # cu_seqlens_k
        cu_seqlens_k_new,
        None,  # seqused_q
        cache_seqlens,
        max_seqlen_q,
        None,  # max_seqlen_k
        page_table,
        cache_batch_idx,
        cache_leftpad,
        rotary_cos,
        rotary_sin,
        rotary_seqlens,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        attention_chunk,
        softcap,
        rotary_interleaved,
        scheduler_metadata,
        num_splits,
        pack_gqa,
        sm_margin,
        sinks,
    )
    # return (out, softmax_lse) if return_softmax_lse else out
    if SGLANG_PROFILE_NVTX:
        th_nvtx_range_pop()

    return (out, softmax_lse, *rest) if return_softmax_lse else out


@maybe_wrap_debug_kernel
def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,
    max_seqlen_k=None,
    seqused_q=None,
    seqused_k=None,
    page_table=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
    sinks=None,
    score_mod=None,
    aux_tensors=None,
    ver=3,
    out=None,
):

    if not is_fa3_supported():
        raise NotImplementedError(
            "flash_attn at sgl-kernel is only supported on sm90 and above"
        )

    # FA3 requires max_seqlen_q and max_seqlen_k
    if max_seqlen_q is None or max_seqlen_k is None:
        raise ValueError("max_seqlen_q and max_seqlen_k are required for FA3")

    if softmax_scale is None:
        softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (
            -0.5
        )
    attention_chunk = 0 if attention_chunk is None else int(attention_chunk)

    if SGLANG_PROFILE_NVTX:
        if torch.cuda.is_current_stream_capturing():
            nvtx_message = f"[FW_FMHA] --format=flash_attn_{ver},Forward,type:D,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_value:{v.shape[-1]},num_heads_k:{k.shape[-2]},num_heads:{q.shape[-2]},batch_size:{len(cu_seqlens_q) - 1},seqlen_k:{max_seqlen_k},data_type:{q.dtype},window_size_left:{window_size[0]},window_size_right:{window_size[1]},softcap:{softcap}"
        else:
            if SGLANG_PROFILE_NVTX_PRINT_SEQLEN:
                cu_seqlens_q_list = (
                    cu_seqlens_q.flatten().cpu().tolist()
                    if cu_seqlens_q is not None
                    else "[]"
                )
                cu_seqlens_k_list = (
                    cu_seqlens_k.flatten().cpu().tolist()
                    if cu_seqlens_k is not None
                    else "[]"
                )
                nvtx_message = f"[FW_FMHA] --format=flash_attn_{ver},Forward,type:P,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_value:{v.shape[-1]},num_heads_k:{k.shape[-2]},num_heads:{q.shape[-2]},batch_size:{len(cu_seqlens_q) - 1},seqlen_k:{max_seqlen_k},data_type:{q.dtype},window_size_left:{window_size[0]},window_size_right:{window_size[1]},softcap:{softcap},cu_seqlens_q:{cu_seqlens_q_list},cu_seqlens_k:{cu_seqlens_k_list}"
            else:
                nvtx_message = f"[FW_FMHA] --format=flash_attn_{ver},Forward,type:P,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_value:{v.shape[-1]},num_heads_k:{k.shape[-2]},num_heads:{q.shape[-2]},batch_size:{len(cu_seqlens_q) - 1},seqlen_k:{max_seqlen_k},data_type:{q.dtype},window_size_left:{window_size[0]},window_size_right:{window_size[1]},softcap:{softcap}"
        th_nvtx_range_push(nvtx_message)

    out, softmax_lse, *rest = torch.ops.sgl_kernel.fwd.default(
        q,
        k,
        v,
        None,  # k_new
        None,  # v_new
        qv,  # qv
        out,  # out
        cu_seqlens_q,
        cu_seqlens_k,
        None,  # cu_seqlens_k_new
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        None,  # page_table,
        None,  # kv_batch_idx
        None,  # leftpad_k
        None,  # rotary cos
        None,  # rotary sin
        None,  # seqlens_rotary
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        attention_chunk,
        softcap,
        is_rotary_interleaved=False,
        scheduler_metadata=None,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
        sinks=sinks,
    )

    if SGLANG_PROFILE_NVTX:
        th_nvtx_range_pop()
    return (out, softmax_lse, *rest) if return_softmax_lse else out


def get_scheduler_metadata(
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads: int,
    num_heads_k: int,
    headdim: int,
    cache_seqlens: torch.Tensor,
    qkv_dtype=torch.bfloat16,
    headdim_v: Optional[int] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    max_seqlen_k_new: int = 0,
    causal: bool = False,
    window_size=(-1, -1),
    attention_chunk: int = 0,
    has_softcap: bool = False,
    num_splits: int = 0,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
):
    """Precompute FA3 tile scheduling metadata.

    Call this once per batch (not per layer) and pass the result as
    scheduler_metadata to flash_attn_with_kvcache / flash_attn_varlen_func.
    This avoids the prepare_varlen_num_blocks kernel running on every layer.
    """
    cache_seqlens = maybe_contiguous(cache_seqlens)
    if headdim_v is None:
        headdim_v = headdim

    if is_ppu():
        return get_metadata(
            batch_size,
            max_seqlen_q,
            max_seqlen_k,
            num_heads,
            num_heads_k,
            headdim,
            cache_seqlens,
            qkv_dtype,
            headdim_v,
            cu_seqlens_q,
            cu_seqlens_k_new,
            leftpad_k,
            page_size,
            max_seqlen_k_new,
            causal,
            window_size,
            attention_chunk,
            has_softcap,
            num_splits,
            pack_gqa,
            sm_margin,
        )

    return torch.ops.sgl_kernel.get_scheduler_metadata(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads,
        num_heads_k,
        headdim,
        headdim_v,
        qkv_dtype,
        cache_seqlens,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_k_new,
        seqused_q,
        leftpad_k,
        page_size,
        max_seqlen_k_new,
        causal,
        window_size[0],
        window_size[1],
        attention_chunk,
        has_softcap,
        num_splits,
        pack_gqa,
        sm_margin,
    )


@maybe_wrap_debug_kernel
def flash_attn_with_kvcache_sail(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    attention_chunk: Optional[int] = None,
    softcap=0.0,  # 0.0 means deactivated
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,  # Can be tuned for speed
    pack_gqa=None,  # Can be tuned for speed
    sm_margin=0,  # Can be tuned if some SMs are used for communication
    return_softmax_lse=False,
    sinks=None,
    score_mod=None,
    aux_tensors=None,
    ver=3,
    out=None,
    max_seqlen_k: Optional[int] = None,
):
    if ver == 4:
        raise NotImplementedError(
            "FA4 is not available, no implemented flash_attn_with_kvcache"
        )
    if SGLANG_PROFILE_NVTX:
        if cu_seqlens_q is not None:
            batch_size = len(cu_seqlens_q) - 1
        else:
            batch_size = 0
        if page_table is not None:
            num_blocks = k_cache.shape[-4]
            page_block_size = k_cache.shape[-3]
            page_table_flag = 1
        else:
            num_blocks = 0
            page_block_size = 0
            page_table_flag = 0

        if torch.cuda.is_current_stream_capturing():
            nvtx_message = f"[FW_FMHA] --format=flash_attn_{ver},Forward,type:D,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_value:{k_cache.shape[-1]},num_heads_k:{k_cache.shape[-2]},num_heads:{q.shape[-2]},batch_size:{batch_size},data_type:{q.dtype},window_size_left:{window_size[0]},window_size_right:{window_size[1]},softcap:{softcap},paged_kv:{page_table_flag}"
        else:
            if SGLANG_PROFILE_NVTX_PRINT_SEQLEN:
                cu_seqlens_q_list = (
                    cu_seqlens_q.flatten().cpu().tolist()
                    if cu_seqlens_q is not None
                    else "[]"
                )
                nvtx_message = f"[FW_FMHA] --format=flash_attn_{ver},Forward,type:P,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_value:{k_cache.shape[-1]},num_heads_k:{k_cache.shape[-2]},num_heads:{q.shape[-2]},batch_size:{batch_size},data_type:{q.dtype},window_size_left:{window_size[0]},window_size_right:{window_size[1]},softcap:{softcap},paged_kv:{page_table_flag},num_blocks:{num_blocks},page_block_size:{page_block_size},cu_seqlens_q:{cu_seqlens_q_list}"
            else:
                nvtx_message = f"[FW_FMHA] --format=flash_attn_{ver},Forward,type:P,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_value:{k_cache.shape[-1]},num_heads_k:{k_cache.shape[-2]},num_heads:{q.shape[-2]},batch_size:{batch_size},data_type:{q.dtype},window_size_left:{window_size[0]},window_size_right:{window_size[1]},softcap:{softcap},paged_kv:{page_table_flag}"
        th_nvtx_range_push(nvtx_message)

    attention_chunk = 0 if attention_chunk is None else int(attention_chunk)

    result = fa_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k,
        v=v,
        qv=qv,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        cache_leftpad=cache_leftpad,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k_new,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        rotary_seqlens=rotary_seqlens,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        attention_chunk=attention_chunk,  # attention_chunk
        softcap=softcap,
        rotary_interleaved=rotary_interleaved,
        scheduler_metadata=scheduler_metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
        return_softmax_lse=return_softmax_lse,
        s_aux=sinks,
    )
    if SGLANG_PROFILE_NVTX:
        th_nvtx_range_pop()

    return result


@maybe_wrap_debug_kernel
def flash_attn_varlen_func_sail(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,
    max_seqlen_k=None,
    seqused_q=None,
    seqused_k=None,
    page_table=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
    sinks=None,
    score_mod=None,
    aux_tensors=None,
    ver=3,
    out=None,
    deterministic=False,
):
    if ver == 4:
        raise NotImplementedError(
            "FA4 is not available, no implemented flash_attn_varlen_func"
        )

    if softmax_scale is None:
        softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (
            -0.5
        )

    attention_chunk = 0 if attention_chunk is None else int(attention_chunk)

    if SGLANG_PROFILE_NVTX:
        if torch.cuda.is_current_stream_capturing():
            nvtx_message = f"[FW_FMHA] --format=flash_attn_{ver},Forward,type:D,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_value:{v.shape[-1]},num_heads_k:{k.shape[-2]},num_heads:{q.shape[-2]},batch_size:{len(cu_seqlens_q) - 1},seqlen_k:{max_seqlen_k},data_type:{q.dtype},window_size_left:{window_size[0]},window_size_right:{window_size[1]},softcap:{softcap}"
        else:
            if SGLANG_PROFILE_NVTX_PRINT_SEQLEN:
                cu_seqlens_q_list = (
                    cu_seqlens_q.flatten().cpu().tolist()
                    if cu_seqlens_q is not None
                    else "[]"
                )
                cu_seqlens_k_list = (
                    cu_seqlens_k.flatten().cpu().tolist()
                    if cu_seqlens_k is not None
                    else "[]"
                )
                nvtx_message = f"[FW_FMHA] --format=flash_attn_{ver},Forward,type:P,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_value:{v.shape[-1]},num_heads_k:{k.shape[-2]},num_heads:{q.shape[-2]},batch_size:{len(cu_seqlens_q) - 1},seqlen_k:{max_seqlen_k},data_type:{q.dtype},window_size_left:{window_size[0]},window_size_right:{window_size[1]},softcap:{softcap},cu_seqlens_q:{cu_seqlens_q_list},cu_seqlens_k:{cu_seqlens_k_list}"
            else:
                nvtx_message = f"[FW_FMHA] --format=flash_attn_{ver},Forward,type:P,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_value:{v.shape[-1]},num_heads_k:{k.shape[-2]},num_heads:{q.shape[-2]},batch_size:{len(cu_seqlens_q) - 1},seqlen_k:{max_seqlen_k},data_type:{q.dtype},window_size_left:{window_size[0]},window_size_right:{window_size[1]},softcap:{softcap}"
        th_nvtx_range_push(nvtx_message)
    output = fa_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        softmax_scale=softmax_scale,
        causal=causal,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        window_size=window_size,
        attention_chunk=attention_chunk,  # attention_chunk
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        deterministic=deterministic,
        sm_margin=sm_margin,
        return_attn_probs=return_softmax_lse,
        s_aux=sinks,
    )

    if SGLANG_PROFILE_NVTX:
        th_nvtx_range_pop()
    return output


# <NOTE>
# Dispatch flash attention implementation
# </NOTE>

if is_ppu():
    flash_attn_with_kvcache = flash_attn_with_kvcache_sail
    flash_attn_varlen_func = flash_attn_varlen_func_sail
