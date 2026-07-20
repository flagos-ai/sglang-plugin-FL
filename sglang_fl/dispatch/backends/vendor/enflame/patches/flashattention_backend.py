from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.layers.utils.cp_utils import (
    cp_allgather_and_save_kv_cache,
    cp_attn_forward_extend,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

from sglang.srt.layers.attention.attention_registry import register_attention_backend

from sgl_kernel import merge_state_v2

from sglang.srt.layers.attention.flashattention_backend import FlashAttentionMetadata

from flash_attn.vllm_flash_attn import (
        flash_attn_varlen_func,
        flash_attn_with_kvcache
    )

def __init__(
    self,
    model_runner: ModelRunner,
    skip_prefill: bool = False,
    speculative_step_id=0,
    topk=0,
    speculative_num_steps=0,
    fa_impl_ver=3,
):
    # super().__init__()

    assert not (
        model_runner.sliding_window_size is not None
        and model_runner.model_config.is_encoder_decoder
    ), "Sliding window and cross attention are not supported together"

    self.is_encoder_decoder = model_runner.model_config.is_encoder_decoder
    self.forward_metadata: FlashAttentionMetadata = None
    # extra metadata for handling speculative decoding topk > 1, extended draft decode and verify
    self.forward_metadata_spec_decode_expand: FlashAttentionMetadata = None
    self.max_context_len = model_runner.model_config.context_len
    self.device = model_runner.device
    self.decode_cuda_graph_metadata = {}
    self.target_verify_metadata = {}
    self.req_to_token = model_runner.req_to_token_pool.req_to_token
    self.kv_cache_dtype = model_runner.kv_cache_dtype
    self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype
    self.page_size = model_runner.page_size
    self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
    self.skip_prefill = skip_prefill
    self.attn_cp_size = model_runner.attn_cp_size

    self.use_sliding_window_kv_pool = (
        isinstance(model_runner.token_to_kv_pool, SWAKVPool)
        and model_runner.token_to_kv_pool.swa_layer_nums > 0
    )
    if self.use_sliding_window_kv_pool:
        self.token_to_kv_pool = model_runner.token_to_kv_pool

    self.topk = model_runner.server_args.speculative_eagle_topk or 0
    self.speculative_num_steps = speculative_num_steps
    self.speculative_num_draft_tokens = (
        model_runner.server_args.speculative_num_draft_tokens
    )
    self.speculative_step_id = speculative_step_id

    # Local attention settings
    self.has_local_attention = model_runner.model_config.is_local_attention_model
    if self.has_local_attention:
        assert (
            model_runner.attention_chunk_size is not None
        ), "Attention chunk size is required for local attention"
        self.attention_chunk_size = model_runner.attention_chunk_size

    # For each layer, the sliding_window_size can be different. This is only used for preparing SWA metadata.
    # We use `layer.sliding_window_size` to decide whether to use SWA for each layer.
    self.sliding_window_size = model_runner.sliding_window_size
    self.has_swa = (
        self.sliding_window_size is not None and self.sliding_window_size > -1
    )

    # Select version
    self.fa_impl_ver = fa_impl_ver
    if self.fa_impl_ver == 3:
        # gcu replace
        from flash_attn.vllm_flash_attn import (
            flash_attn_varlen_func,
            flash_attn_with_kvcache,
            get_scheduler_metadata,
        )

        self._get_scheduler_metadata = get_scheduler_metadata
    elif self.fa_impl_ver == 4:
        from sglang.jit_kernel.flash_attention_v4 import (
            flash_attn_varlen_func,
            flash_attn_with_kvcache,
        )

        self._get_scheduler_metadata = None
    else:
        raise ValueError(f"Invalid version: {self.fa_impl_ver=}")

    self.flash_attn_varlen_func = flash_attn_varlen_func
    self.flash_attn_with_kvcache = flash_attn_with_kvcache

    # Store head info for precomputing FA3 scheduler metadata
    self.head_dim = model_runner.model_config.head_dim
    self.num_attention_heads = (
        model_runner.model_config.hf_text_config.num_attention_heads
        // model_runner.tp_size
    )
    self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
        model_runner.tp_size
    )
    _softcapping = getattr(
        model_runner.model_config.hf_text_config, "attn_logit_softcapping", None
    )
    self.has_softcap = _softcapping is not None and _softcapping > 0.0

    # If num_splits == 0, we use a heuristic to automatically determine the number of splits.
    # We set nums splits to 1 if deterministic inference is enabled.
    # See https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ for more details.
    # Furthermore, FA4 does not support num_splits=0 with CUDA Graph, so we set num_splits to 1 if CUDA Graph is enabled.
    self.num_splits = (
        1
        if model_runner.server_args.enable_deterministic_inference
        or (
            self.fa_impl_ver == 4
            and not model_runner.server_args.disable_cuda_graph
        )
        else 0
    )

    # In embedding mode with no chunked prefill and radix cache disabled,
    # skip KV cache write and use flash_attn_varlen_func with raw K/V
    # instead of flash_attn_with_kvcache, bypassing paged KV cache entirely.
    server_args = model_runner.server_args
    self.fa_skip_kv_cache = (
        server_args.is_embedding
        and server_args.chunked_prefill_size == -1
        and server_args.disable_radix_cache
    )

def _compute_scheduler_metadata(
    self, batch_size, max_seq_len_k, cache_seqlens, cu_seqlens_q
):
    """Compute FA3 scheduler metadata for decode.

    Returns the scheduler_metadata tensor, or None if not applicable.
    """
    if self._get_scheduler_metadata is None or self.use_mla:
        return None
    # Always use window_size=(-1, -1) because scheduler_metadata is only
    # consumed by non-SWA layers (SWA layers skip it in forward_decode).
    return self._get_scheduler_metadata(
        batch_size=batch_size,
        max_seqlen_q=1,
        max_seqlen_k=max_seq_len_k,
        num_heads_q=self.num_attention_heads, # gcu replace
        num_heads_kv=self.num_kv_heads, # gcu replace
        headdim=self.head_dim,
        cache_seqlens=cache_seqlens,
        qkv_dtype=self.kv_cache_dtype,
        cu_seqlens_q=cu_seqlens_q,
        page_size=self.page_size,
        causal=True,
        has_softcap=self.has_softcap,
        num_splits=self.num_splits,
    )

def forward_extend(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    save_kv_cache=True,
    # For multi-head latent attention
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
):
    if k is not None:
        assert v is not None

        is_cp_mode = (
            forward_batch.forward_mode.is_context_parallel_extend()
            and forward_batch.attn_cp_metadata is not None
            and self.attn_cp_size > 1
        )

        if save_kv_cache and not is_cp_mode and not self.fa_skip_kv_cache:
            cache_loc = (
                forward_batch.out_cache_loc
                if not layer.is_cross_attention
                else forward_batch.encoder_out_cache_loc
            )
            if not self.use_mla:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )
            else:
                forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                    layer,
                    cache_loc,
                    k,
                    k_rope,
                )
        if is_cp_mode:
            cp_allgather_and_save_kv_cache(
                forward_batch, layer, k, v, self.attn_cp_size
            )

    # Use precomputed metadata across all layers
    metadata = self.forward_metadata

    # Calculate window size (can be moved to metadata if layer properties don't change)
    # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
    # here is two side inclusive
    is_swa_layer = (
        layer.sliding_window_size is not None and layer.sliding_window_size > -1
    )
    window_size = (layer.sliding_window_size, 0) if is_swa_layer else (-1, -1)
    k_descale, v_descale = None, None
    # only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
    # has corresponding quantization method so that layer.k_scale is not None,
    # 3) layer.head_dim <= 256 since fa3 kernel require fp16 and bf16 data type in this case,
    # 4) fa_impl_ver != 4 since fa4 does not currently support fp8 queries and keys.
    if (
        self.kv_cache_dtype_str != "auto"
        and layer.head_dim <= 256
        and self.fa_impl_ver != 4
    ):
        if layer.k_scale is not None:
            descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
            k_descale = layer.k_scale.expand(descale_shape)
            v_descale = layer.v_scale.expand(descale_shape)
        q = q.to(self.kv_cache_dtype)
        q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
        k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
    causal = True
    if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
        causal = False

    # Check if we should use local attention
    use_local_attn = (
        self.has_local_attention
        and self.attention_chunk_size is not None
        and metadata.local_attn_metadata is not None
        and (hasattr(layer, "use_irope") and layer.use_irope)
    )

    # We do cascade attention for Target Verify with topk > 1
    # We don't use cascade attention for Sliding Window Attention:
    # - Different window sizes should be passed in for each q in the first stage of cascade attention, but FA3 interface doesn't support pass in a list of window sizes.
    # - The overhead of duplicated computation of the common prefix part is small for sliding window layers (seq_len <= window_size), so we can just expand it.
    use_cascade_attn = (
        forward_batch.forward_mode.is_target_verify()
        and self.topk > 1
        and not is_swa_layer
    )

    kwargs = {}
    if sinks is not None:
        kwargs["sinks"] = sinks

    _fa_out = (
        forward_batch._attn_output.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        if getattr(forward_batch, "_attn_output", None) is not None
        else None
    )

    # Get the appropriate page table based on whether we're using local attention
    if use_local_attn:
        local_metadata = metadata.local_attn_metadata
        page_table = local_metadata.local_block_table
        cu_seqlens_q = local_metadata.local_query_start_loc
        cache_seqlens = local_metadata.local_seqused_k
        max_seqlen_q = local_metadata.local_max_query_len
    elif is_swa_layer and metadata.swa_spec_metadata is not None:
        swa_spec_metadata = metadata.swa_spec_metadata
        page_table = swa_spec_metadata.page_table
        cu_seqlens_q = swa_spec_metadata.cu_seqlens_q
        cache_seqlens = swa_spec_metadata.cache_seqlens_int32
        max_seqlen_q = swa_spec_metadata.max_seq_len_q
        cu_seqlens_k = swa_spec_metadata.cu_seqlens_k
    else:
        page_table = metadata.page_table
        if is_swa_layer and self.use_sliding_window_kv_pool:
            if metadata.swa_page_table is not None:
                page_table = metadata.swa_page_table
            else:
                page_table = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    metadata.page_table
                )
        cu_seqlens_q = metadata.cu_seqlens_q
        cache_seqlens = metadata.cache_seqlens_int32
        max_seqlen_q = metadata.max_seq_len_q
        cu_seqlens_k = metadata.cu_seqlens_k

    # Use Flash Attention for prefill
    if not self.use_mla:
        # Do multi-head attention
        key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )

        key_cache = key_cache.view(
            -1, self.page_size, layer.tp_k_head_num, layer.head_dim
        )
        value_cache = value_cache.view(
            -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
        )
        if layer.is_cross_attention:
            page_table = metadata.encoder_page_table
            cache_seqlens = metadata.encoder_lens_int32
            cu_seqlens_k = metadata.encoder_cu_seqlens_k
            window_size = (-1, -1)

        if (
            forward_batch.forward_mode.is_context_parallel_extend()
            and forward_batch.attn_cp_metadata is not None
            and self.attn_cp_size > 1
        ):

            def _fa_cp_attn(
                q_chunk, cu_seqlens_q_cp, cache_seqlens_cp, max_seqlen_q_cp
            ):
                return flash_attn_with_kvcache(
                    q=q_chunk,
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens_cp,
                    cu_seqlens_q=cu_seqlens_q_cp,
                    cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                    max_seqlen_q=max_seqlen_q_cp,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                    num_splits=self.num_splits,
                    ver=self.fa_impl_ver,
                    **kwargs,
                )

            result = cp_attn_forward_extend(
                forward_batch,
                q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                self.device,
                _fa_cp_attn,
            )
        elif self.fa_skip_kv_cache:
            # Embedding mode: skip KV cache read and use raw K/V tensors
            # directly via flash_attn_varlen_func. The KV cache write is
            # also skipped (guarded above). This eliminates store_kvcache
            # and prepare_varlen_num_blocks overhead per layer.
            assert k is not None, "fa_skip_kv_cache requires k to be provided"
            assert k_descale is None and v_descale is None, (
                "fa_skip_kv_cache uses raw K/V tensors, "
                "FP8 KV cache descaling is not supported in this mode"
            )
            result = flash_attn_varlen_func(
                q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                k=k.view(-1, layer.tp_k_head_num, layer.head_dim),
                v=v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_q,
                softmax_scale=layer.scaling,
                causal=causal,
                window_size=window_size,
                softcap=layer.logit_cap,
                num_splits=self.num_splits,
                out=_fa_out,
                **kwargs,
            )
        else:
            # result = flash_attn_with_kvcache(
            #     q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            #     k_cache=key_cache,
            #     v_cache=value_cache,
            #     page_table=page_table,
            #     cache_seqlens=cache_seqlens,
            #     cu_seqlens_q=cu_seqlens_q,
            #     cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
            #     max_seqlen_q=max_seqlen_q,
            #     softmax_scale=layer.scaling,
            #     causal=False if use_cascade_attn else causal,
            #     window_size=window_size,
            #     softcap=layer.logit_cap,
            #     k_descale=k_descale,
            #     v_descale=v_descale,
            #     return_softmax_lse=use_cascade_attn,
            #     num_splits=self.num_splits,
            #     out=_fa_out,
            #     ver=self.fa_impl_ver,
            #     **kwargs,
            # )

            # gcu replace begin
            result = flash_attn_varlen_func(
                q=q.reshape(-1, layer.tp_q_head_num, layer.head_dim), #[:real_token_num]
                k=key_cache,
                v=value_cache,
                # out=result[:real_token_num],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=cache_seqlens,
                max_seqlen_k=self.max_context_len, #metadata.max_seq_len_k,
                softmax_scale=layer.scaling,
                causal=False if use_cascade_attn else causal,
                # alibi_slopes=self.alibi_slopes,
                window_size=window_size,
                block_table=page_table,
                softcap=layer.logit_cap,
                # scheduler_metadata=scheduler_metadata,
                fa_version=3,
                # q_descale=layer._q_scale.expand(descale_shape),
                # k_descale=layer._k_scale.expand(descale_shape),
                # v_descale=layer._v_scale.expand(descale_shape),
                num_splits=self.num_splits,
                s_aux=sinks
            )
            ###### A workaround to fix mimo-v2-flash mtp3 gsm8k_cot precision bug START
            ###### The root cause is not yet fully clear and requires further investigation
            seq_len2 = cu_seqlens_q[-1]
            indices2 = torch.arange(result.size(0), device=result.device)
            valid_mask2 = (indices2 < seq_len2).view(-1, 1, 1)
            result = torch.where(valid_mask2, result, 0.0)
            ###### The workaround to fix mimo-v2-flash mtp3 gsm8k_cot precision bug END

            # gcu replace end

        if use_cascade_attn:
            o, softmax_lse, *rest = result
            o_expand, softmax_lse_expand, *rest_expand = flash_attn_with_kvcache(
                q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                # Here metadata_expand.page_table is not divided with page_size.
                # This is because we loose the fine control of  what token to attend,
                # but has to attend to some block completely.
                k_cache=key_cache.view(-1, 1, layer.tp_k_head_num, layer.head_dim),
                v_cache=value_cache.view(
                    -1, 1, layer.tp_v_head_num, layer.head_dim
                ),
                page_table=self.forward_metadata_spec_decode_expand.page_table,
                cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                softmax_scale=layer.scaling,
                causal=False,
                window_size=window_size,
                softcap=layer.logit_cap,
                k_descale=k_descale,
                v_descale=v_descale,
                return_softmax_lse=True,
                num_splits=self.num_splits,
                ver=self.fa_impl_ver,
                **kwargs,
            )
            o, _ = merge_state_v2_wrapper(
                o,
                softmax_lse.T.contiguous(),
                o_expand,
                softmax_lse_expand.T.contiguous(),
            )
        else:
            o = result
    else:
        if (
            forward_batch.attn_attend_prefix_cache is not None
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
        ):
            # Do multi-head attention with chunked prefix cache
            if forward_batch.attn_attend_prefix_cache:
                assert not get_global_server_args().disable_chunked_prefix_cache
                # MHA for chunked prefix kv cache when running model with MLA
                assert forward_batch.prefix_chunk_idx is not None
                assert forward_batch.prefix_chunk_cu_seq_lens is not None
                assert forward_batch.prefix_chunk_max_seq_lens is not None

                chunk_idx = forward_batch.prefix_chunk_idx
                assert chunk_idx >= 0

                assert forward_batch.mha_return_lse
                output = flash_attn_varlen_func(
                    q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                    v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k=forward_batch.prefix_chunk_cu_seq_lens[chunk_idx],
                    max_seqlen_q=metadata.max_seq_len_q,
                    max_seqlen_k=forward_batch.prefix_chunk_max_seq_lens[chunk_idx],
                    softmax_scale=layer.scaling,
                    causal=False,
                    return_softmax_lse=True,
                    out=_fa_out,
                    ver=self.fa_impl_ver,
                    **kwargs,
                )
            else:
                # MHA for extend part of sequence without attending prefix kv cache
                cu_seqlens_k = (
                    metadata.cu_seqlens_q
                    if not forward_batch.mha_one_shot
                    else metadata.cu_seqlens_k
                )
                max_seqlen_k = (
                    metadata.max_seq_len_q
                    if not forward_batch.mha_one_shot
                    else metadata.max_seq_len_k
                )
                output = flash_attn_varlen_func(
                    q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                    v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=metadata.max_seq_len_q,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=layer.scaling,
                    causal=True,
                    return_softmax_lse=forward_batch.mha_return_lse,
                    out=_fa_out,
                    ver=self.fa_impl_ver,
                    **kwargs,
                )
            if forward_batch.mha_return_lse:
                output, lse, *rest = output
                lse = torch.transpose(lse, 0, 1).contiguous()
                return output, lse
            return output
        else:
            assert self.fa_impl_ver == 3, "Only FA3 support here"
            # Do absorbed multi-latent attention
            kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                layer.layer_id
            ).to(q.dtype)
            k_rope = kv_cache[:, :, layer.v_head_dim :]
            c_kv = kv_cache[:, :, : layer.v_head_dim]
            k_rope_cache = k_rope.view(
                -1,
                self.page_size,
                layer.tp_k_head_num,
                layer.head_dim - layer.v_head_dim,
            )
            c_kv_cache = c_kv.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )
            if q_rope is not None:
                q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                q_rope = q_rope.view(
                    -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                )
            else:
                q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                q_nope = q_all[:, :, : layer.v_head_dim]
                q_rope = q_all[:, :, layer.v_head_dim :]

            result = flash_attn_with_kvcache(
                q=q_rope,
                k_cache=k_rope_cache,
                v_cache=c_kv_cache,
                qv=q_nope,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=layer.scaling,
                causal=False if use_cascade_attn else causal,
                softcap=layer.logit_cap,
                k_descale=k_descale,
                v_descale=v_descale,
                return_softmax_lse=use_cascade_attn,
                num_splits=self.num_splits,
                ver=self.fa_impl_ver,
            )
            if use_cascade_attn:
                o, softmax_lse, *rest = result
                o_expand, softmax_lse_expand, *rest_expand = (
                    flash_attn_with_kvcache(
                        q=q_rope,
                        k_cache=k_rope_cache,
                        v_cache=c_kv_cache,
                        qv=q_nope,
                        page_table=self.forward_metadata_spec_decode_expand.page_table,
                        cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                        cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                        cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                        max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                        softmax_scale=layer.scaling,
                        causal=False,
                        window_size=window_size,
                        softcap=layer.logit_cap,
                        k_descale=k_descale,
                        v_descale=v_descale,
                        return_softmax_lse=True,
                        num_splits=self.num_splits,
                        ver=self.fa_impl_ver,
                    )
                )
                o, _ = merge_state_v2_wrapper(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                )
            else:
                o = result

    return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

def forward_decode(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    save_kv_cache=True,
    # For multi-head latent attention
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if k is not None:
        assert v is not None
        if save_kv_cache:
            cache_loc = (
                forward_batch.out_cache_loc
                if not layer.is_cross_attention
                else forward_batch.encoder_out_cache_loc
            )
            if not self.use_mla:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )
            else:
                forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                    layer,
                    cache_loc,
                    k,
                    k_rope,
                )

    # Use precomputed metadata across all layers
    metadata = self.forward_metadata
    local_attn_metadata = getattr(metadata, "local_attn_metadata", None)
    use_local_attn = (
        self.has_local_attention
        and self.attention_chunk_size is not None
        and local_attn_metadata is not None
        and (hasattr(layer, "use_irope") and layer.use_irope)
    )

    # When Spec Decode enabled, forward_decode would be called with two mode:
    # 1. DRAFT_DECODE: we enable cascade attention when top_k > 1
    # 2. IDLE: we don’t need cascade attention, spec_info will be none in this case
    use_cascade_attn = forward_batch.spec_info is not None and self.topk > 1

    # Calculate window size (can be moved to metadata if layer properties don't change)
    # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
    # here is two side inclusive
    is_swa_layer = (
        layer.sliding_window_size is not None and layer.sliding_window_size > -1
    )
    window_size = (layer.sliding_window_size, 0) if is_swa_layer else (-1, -1)

    causal = True
    if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
        causal = False

    kwargs = {}
    if sinks is not None:
        kwargs["sinks"] = sinks

    _fa_out = (
        forward_batch._attn_output.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        if getattr(forward_batch, "_attn_output", None) is not None
        else None
    )

    k_descale, v_descale = None, None
    # only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
    # has corresponding quantization method so that layer.k_scale is not None,
    # 3) layer.head_dim <= 256 since fa3 kernel require fp16 and bf16 data type in this case.
    if self.kv_cache_dtype_str != "auto" and layer.head_dim <= 256:
        if layer.k_scale is not None:
            descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
            k_descale = layer.k_scale.expand(descale_shape)
            v_descale = layer.v_scale.expand(descale_shape)
        q = q.to(self.kv_cache_dtype)
        q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
        k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
    if not self.use_mla:
        # Do multi-head attention

        key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )
        key_cache = key_cache.view(
            -1, self.page_size, layer.tp_k_head_num, layer.head_dim
        )
        value_cache = value_cache.view(
            -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
        )

        if layer.is_cross_attention:
            # Always use non-chunked logic for cross-attention
            o = flash_attn_with_kvcache(
                q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                k_cache=key_cache,
                v_cache=value_cache,
                page_table=metadata.encoder_page_table,
                cache_seqlens=metadata.encoder_lens_int32,
                cu_seqlens_q=metadata.cu_seqlens_q,
                cu_seqlens_k_new=metadata.encoder_cu_seqlens_k,
                max_seqlen_q=1,
                softmax_scale=layer.scaling,
                causal=False,
                window_size=(-1, -1),
                softcap=layer.logit_cap,
                k_descale=k_descale,
                v_descale=v_descale,
                num_splits=self.num_splits,
                ver=self.fa_impl_ver,
                **kwargs,
            )
        elif use_local_attn:
            # Use chunked (local) attention batching for self-attention
            o = flash_attn_with_kvcache(
                q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                k_cache=key_cache,
                v_cache=value_cache,
                page_table=local_attn_metadata.local_block_table,
                cache_seqlens=local_attn_metadata.local_seqused_k,
                cu_seqlens_q=local_attn_metadata.local_query_start_loc,
                cu_seqlens_k_new=None,
                max_seqlen_q=local_attn_metadata.local_max_query_len,
                softmax_scale=layer.scaling,
                causal=True,
                window_size=(-1, -1),
                softcap=layer.logit_cap,
                k_descale=k_descale,
                v_descale=v_descale,
                num_splits=self.num_splits,
                ver=self.fa_impl_ver,
                **kwargs,
            )
        else:
            page_table = metadata.page_table
            if is_swa_layer and self.use_sliding_window_kv_pool:
                if metadata.swa_page_table is not None:
                    page_table = metadata.swa_page_table
                else:
                    page_table = (
                        self.token_to_kv_pool.translate_loc_from_full_to_swa(
                            metadata.page_table
                        )
                    )
            cache_seqlens = metadata.cache_seqlens_int32
            cu_seqlens_k = metadata.cu_seqlens_k
            max_seqlen_q = metadata.max_seq_len_q
            q_reshaped = q.contiguous().view(
                -1, layer.tp_q_head_num, layer.head_dim
            )

            # Default: single-token self-attention
            # Use precomputed scheduler_metadata when available and applicable.
            # scheduler_metadata is only valid for non-SWA, non-cascade decode.
            sched_meta = None
            if (
                metadata.scheduler_metadata is not None
                and not is_swa_layer
                and not use_cascade_attn
            ):
                sched_meta = metadata.scheduler_metadata
            # result = flash_attn_with_kvcache(
            #     q=q_reshaped,
            #     k_cache=key_cache,
            #     v_cache=value_cache,
            #     page_table=page_table,
            #     cache_seqlens=cache_seqlens,
            #     cu_seqlens_q=metadata.cu_seqlens_q,
            #     max_seqlen_q=max_seqlen_q,
            #     softmax_scale=layer.scaling,
            #     causal=False if use_cascade_attn else causal,
            #     window_size=window_size,
            #     softcap=layer.logit_cap,
            #     k_descale=k_descale,
            #     v_descale=v_descale,
            #     return_softmax_lse=use_cascade_attn,
            #     num_splits=self.num_splits,
            #     out=_fa_out,
            #     ver=self.fa_impl_ver,
            #     scheduler_metadata=sched_meta,
            #     **kwargs,
            # )

            # gcu replace start
            result = flash_attn_varlen_func(
                q=q_reshaped, # [:real_token_num]
                k=key_cache,
                v=value_cache,
                # out=result[:real_token_num],
                cu_seqlens_q=metadata.cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=cache_seqlens,
                max_seqlen_k=self.max_context_len, #metadata.max_seq_len_k,
                softmax_scale=layer.scaling,
                causal=False if use_cascade_attn else causal,
                # alibi_slopes=self.alibi_slopes,
                window_size=window_size,
                block_table=page_table,
                softcap=layer.logit_cap,
                # scheduler_metadata=scheduler_metadata,
                fa_version=3,
                # q_descale=layer._q_scale.expand(descale_shape),
                # k_descale=layer._k_scale.expand(descale_shape),
                # v_descale=layer._v_scale.expand(descale_shape),
                num_splits=self.num_splits,
                s_aux=sinks
            )
            # gcu replace end
            if use_cascade_attn:
                o, softmax_lse, *rest = result
                o_expand, softmax_lse_expand, *rest_expand = (
                    flash_attn_with_kvcache(
                        q=q_reshaped,
                        k_cache=key_cache,
                        v_cache=value_cache,
                        page_table=self.forward_metadata_spec_decode_expand.page_table,
                        cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                        cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                        cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                        max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                        softmax_scale=layer.scaling,
                        causal=False,
                        window_size=window_size,
                        softcap=layer.logit_cap,
                        k_descale=k_descale,
                        v_descale=v_descale,
                        return_softmax_lse=True,
                        num_splits=self.num_splits,
                        ver=self.fa_impl_ver,
                        **kwargs,
                    )
                )
                o, _ = merge_state_v2(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                )
            else:
                o = result
    else:
        # Do absorbed multi-latent attention
        kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
            q.dtype
        )
        k_rope = kv_cache[:, :, layer.v_head_dim :]
        c_kv = kv_cache[:, :, : layer.v_head_dim]
        k_rope_cache = k_rope.view(
            -1,
            self.page_size,
            layer.tp_k_head_num,
            layer.head_dim - layer.v_head_dim,
        )
        c_kv_cache = c_kv.view(
            -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
        )

        if q_rope is not None:
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
        else:
            q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            q_nope = q_all[:, :, : layer.v_head_dim]
            q_rope = q_all[:, :, layer.v_head_dim :]
        max_seqlen_q = metadata.max_seq_len_q

        result = flash_attn_with_kvcache(
            q=q_rope,
            k_cache=k_rope_cache,
            v_cache=c_kv_cache,
            qv=q_nope,
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens_int32,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k_new=metadata.cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=layer.scaling,
            causal=False if use_cascade_attn else causal,
            softcap=layer.logit_cap,
            k_descale=k_descale,
            v_descale=v_descale,
            return_softmax_lse=use_cascade_attn,  # softmax_lse is needed for merge states
            num_splits=self.num_splits,
            ver=self.fa_impl_ver,
        )
        if use_cascade_attn:
            o, softmax_lse, *rest = result
            o_expand, softmax_lse_expand, *rest_expand = flash_attn_with_kvcache(
                q=q_rope,
                k_cache=k_rope_cache,
                v_cache=c_kv_cache,
                qv=q_nope,
                page_table=self.forward_metadata_spec_decode_expand.page_table,
                cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                softmax_scale=layer.scaling,
                causal=False,
                window_size=window_size,
                softcap=layer.logit_cap,
                k_descale=k_descale,
                v_descale=v_descale,
                return_softmax_lse=True,
                num_splits=self.num_splits,
                ver=self.fa_impl_ver,
            )
            o, _ = merge_state_v2(
                o,
                softmax_lse.T.contiguous(),
                o_expand,
                softmax_lse_expand.T.contiguous(),
            )
        else:
            o = result

    return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

@register_attention_backend("fa3")
def _create_flashattention_v3_backend_hook(runner):
    from sglang.srt.layers.attention.flashattention_backend import (
        FlashAttentionBackend,
    )
    return FlashAttentionBackend(runner)

def patch_flashattention_backend():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    FLASH_ATTN_BACKEND = "sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend"
    HookRegistry.register(f"{FLASH_ATTN_BACKEND}.__init__", __init__, HookType.REPLACE)
    HookRegistry.register(f"{FLASH_ATTN_BACKEND}.forward_extend", forward_extend, HookType.REPLACE)
    HookRegistry.register(f"{FLASH_ATTN_BACKEND}.forward_decode", forward_decode, HookType.REPLACE)
    HookRegistry.register(f"{FLASH_ATTN_BACKEND}._compute_scheduler_metadata", _compute_scheduler_metadata, HookType.REPLACE)

    _TARGET_FUNC = "sglang.srt.layers.attention.attention_registry.create_flashattention_v3_backend"
    HookRegistry.register(f"{_TARGET_FUNC}", _create_flashattention_v3_backend_hook, HookType.REPLACE)
