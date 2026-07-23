
from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    is_dp_attention_enabled,
)

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.utils.hash import murmur_hash32
from sglang.srt.layers.utils.logprob import get_token_ids_logprobs, get_top_logprobs
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils.common import (
    crash_on_warnings,
    get_bool_env_var,
    is_cuda,
    is_npu,
)

if is_cuda():
    import flashinfer
    from flashinfer import (
        min_p_sampling_from_probs,
        top_k_renorm_probs,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_probs,
    )

from sglang.srt.layers.sampler import (
    sampling_from_probs_torch,
    top_k_top_p_min_p_sampling_from_probs_torch,
)

def patch_sampler() -> None:
    module_name = "sglang.srt.layers.sampler"

    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        raise RuntimeError(f"failed to import {module_name}") from error

    def _metax_sample_from_probs(
        self,
        probs: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        positions: torch.Tensor,
        simple_sampling_case: bool,
    ) -> torch.Tensor:
        """Sample from probability distribution (after softmax).

        Used for standard sampling with flashinfer/pytorch backends.
        Handles both simple (direct multinomial) and complex (top-k/top-p/min-p) cases.
        """
        if simple_sampling_case:
            batch_next_token_ids = sampling_from_probs_torch(
                probs,
                sampling_seed=sampling_info.sampling_seed,
                positions=positions,
            )
        else:
            backend = get_global_server_args().sampling_backend
            if backend == "flashinfer":
                assert (
                    sampling_info.sampling_seed is None
                ), "Sampling seed is not supported for flashinfer backend"
                max_top_k_round, batch_size = 32, probs.shape[0]
                uniform_samples = torch.rand(
                    (max_top_k_round, batch_size), device=probs.device
                )
                if sampling_info.need_min_p_sampling:
                    probs = top_k_renorm_probs(probs, sampling_info.top_ks)
                    probs = top_p_renorm_probs(probs, sampling_info.top_ps)
                    batch_next_token_ids = min_p_sampling_from_probs(
                        probs, uniform_samples, sampling_info.min_ps
                    )
                else:
                    batch_next_token_ids, success = top_k_top_p_sampling_from_probs(
                        probs.contiguous(),
                        uniform_samples,
                        sampling_info.top_ks,
                        sampling_info.top_ps,
                        filter_apply_order="joint",
                        check_nan=self.use_nan_detection,
                    )
            elif backend == "pytorch":
                # A slower fallback implementation with torch native operations.
                batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
                    probs,
                    sampling_info.top_ks,
                    sampling_info.top_ps,
                    sampling_info.min_ps,
                    sampling_info.need_min_p_sampling,
                    sampling_info.sampling_seed,
                    positions,
                )
            else:
                raise ValueError(f"Invalid sampling backend: {backend}")

        return batch_next_token_ids

    module.Sampler._sample_from_probs = _metax_sample_from_probs
