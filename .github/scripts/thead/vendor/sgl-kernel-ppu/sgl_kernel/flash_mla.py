from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs

try:
    from sgl_kernel import flashmla_ops  # triggers TORCH extension registration
except Exception as _e:
    _flashmla_import_error = _e
else:
    _flashmla_import_error = None

_IMPORT_ERROR = ImportError(
    "Failed to load sgl_kernel.flashmla_ops extension. Ensure CUDA Driver >= 12.4"
)

from sglang.srt.utils import is_ppu

if is_ppu():
    try:
        import flash_mla as flashmla
        from flash_mla import FlashMLASchedMeta

        _flashmla_import_error = None
        _ppu_flashmla_imported = True
    except Exception as _e:
        _flashmla_import_error = _e
        _ppu_flashmla_imported = False

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


def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
    num_heads_q: Optional[int] = None,
    is_fp8_kvcache: bool = False,
    topk: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        cache_seqlens: (batch_size), dtype torch.int32.
        num_q_tokens_per_head_k: Equals to num_q_tokens_per_q_seq * num_heads_q // num_heads_k.
        num_heads_k: The number of k heads.
        num_heads_q: The number of q heads. This argument is optional when sparse attention is not enabled
        is_fp8_kvcache: Whether the k_cache and v_cache are in fp8 format.
        topk: If not None, sparse attention will be enabled, and only tokens in the `indices` array passed to `flash_mla_with_kvcache_sm90` will be attended to.

    Returns:
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
        num_splits: (batch_size + 1), dtype torch.int32.
    """
    if _flashmla_import_error is not None:
        raise _IMPORT_ERROR from _flashmla_import_error

    if is_fp8_kvcache and topk is None:
        if _ppu_flashmla_imported:
            raise NotImplementedError(
                "FlashMLA dense FP8 get_mla_metadata is not supported."
            )

        return torch.ops.sgl_kernel.get_mla_decoding_metadata_dense_fp8.default(
            cache_seqlens,
            num_q_tokens_per_head_k,
            num_heads_k,
        )

    if _ppu_flashmla_imported:
        return flashmla.get_mla_metadata(
            cache_seqlens,
            num_q_tokens_per_head_k,
            num_heads_k,
            num_heads_q=num_heads_q,
            is_fp8_kvcache=is_fp8_kvcache,
            topk=topk,
        )

    return torch.ops.sgl_kernel.get_mla_decoding_metadata.default(
        cache_seqlens,
        num_q_tokens_per_head_k,
        num_heads_k,
        num_heads_q,
        is_fp8_kvcache,
        topk,
    )


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    descale_q: torch.Tensor | None = None,
    descale_k: torch.Tensor | None = None,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head dimension of v.
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), torch.int32, returned by get_mla_metadata.
        num_splits: (batch_size + 1), torch.int32, returned by get_mla_metadata.
        softmax_scale: float. The scale of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
        causal: bool. Whether to apply causal attention mask.
        descale_q: (batch_size), torch.float32. Descaling factors for Q, used for fp8 quantization.
        descale_k: (batch_size), torch.float32. Descaling factors for K, used for fp8 quantization.
        is_fp8_kvcache: bool. Whether the k_cache and v_cache are in fp8 format. For the format of FP8 KV cache, please refer to README.md
        indices: (batch_size, seq_len_q, topk), torch.int32. If not None, sparse attention will be enabled, and only tokens in the `indices` array will be attended to. Invalid indices should be set to -1 or numbers >= total_seq_len_kv. For details about how to set up `indices`, please refer to README.md.

    Returns:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    if _flashmla_import_error is not None:
        raise _IMPORT_ERROR from _flashmla_import_error

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if indices is not None:
        assert causal == False, "causal must be `false` if sparse attention is enabled."
    assert (descale_q is None) == (
        descale_k is None
    ), "descale_q and descale_k should be both None or both not None"

    if SGLANG_PROFILE_NVTX:
        batch_size = q.shape[0]
        if len(q) == 4:
            max_seqlen_q = q.shape[-3]
        else:
            max_seqlen_q = 1

        if torch.cuda.is_current_stream_capturing():
            nvtx_message = f"[FW_FMHA] --format=MLA,Forward,type:D,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_v:{head_dim_v},num_heads_kv:{k_cache.shape[-2]},num_heads:{q.shape[-2]},batch_size:{batch_size},data_type:{q.dtype},causal:{causal}"
        else:
            if SGLANG_PROFILE_NVTX_PRINT_SEQLEN:
                cu_seqlens_k_list = (
                    cache_seqlens.flatten().cpu().tolist()
                    if cache_seqlens is not None
                    else "[]"
                )
                nvtx_message = f"[FW_FMHA] --format=MLA,Forward,type:P,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_v:{head_dim_v},num_heads_kv:{k_cache.shape[-2]},num_heads:{q.shape[-2]},batch_size:{batch_size},data_type:{q.dtype},causal:{causal},num_blocks:{k_cache.shape[-4]},page_block_size:{k_cache.shape[-3]},cu_seqlens_k:{cu_seqlens_k_list}"
            else:
                nvtx_message = f"[FW_FMHA] --format=MLA,Forward,type:P,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_v:{head_dim_v},num_heads_kv:{k_cache.shape[-2]},num_heads:{q.shape[-2]},batch_size:{batch_size},data_type:{q.dtype},causal:{causal}"
        th_nvtx_range_push(nvtx_message)

    if indices is None and q.element_size() == 1:
        if _ppu_flashmla_imported:
            raise NotImplementedError(
                "FlashMLA dense FP8 with kvcache is not supported."
            )

        out, softmax_lse = torch.ops.sgl_kernel.fwd_kvcache_mla_fp8.default(
            q,
            k_cache,
            head_dim_v,
            cache_seqlens,
            block_table,
            softmax_scale,
            causal,
            tile_scheduler_metadata,
            num_splits,
            descale_q,
            descale_k,
        )
    else:
        if _ppu_flashmla_imported:
            out, softmax_lse = flashmla.flash_mla_with_kvcache(
                q,
                k_cache,
                block_table,
                cache_seqlens,
                head_dim_v,
                tile_scheduler_metadata=FlashMLASchedMeta(
                    tile_scheduler_metadata=tile_scheduler_metadata,
                    num_splits=num_splits,
                ),
                softmax_scale=softmax_scale,
                causal=causal,
                is_fp8_kvcache=is_fp8_kvcache,
                indices=indices,
            )
            if SGLANG_PROFILE_NVTX:
                th_nvtx_range_pop()
            return out, softmax_lse

        out, softmax_lse = torch.ops.sgl_kernel.fwd_kvcache_mla.default(
            q,
            k_cache,
            head_dim_v,
            cache_seqlens,
            block_table,
            softmax_scale,
            causal,
            tile_scheduler_metadata,
            num_splits,
            is_fp8_kvcache,
            indices,
        )
    if SGLANG_PROFILE_NVTX:
        th_nvtx_range_pop()
    return out, softmax_lse


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention prefill kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
        sm_scale: float
        d_v: The dimension of value vectors. Can only be 512

    Returns:
        (output, max_logits, lse)
        About the definition of output, max_logits and lse, please refer to README.md
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, 2-based log-sum-exp
    """
    if _flashmla_import_error is not None:
        raise _IMPORT_ERROR from _flashmla_import_error

    if _ppu_flashmla_imported:
        return flashmla.flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            sm_scale,
            d_v=d_v,
        )

    results = torch.ops.sgl_kernel.sparse_prefill_fwd.default(
        q, kv, indices, sm_scale, d_v
    )
    return results
