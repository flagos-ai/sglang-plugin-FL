import torch

from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

def write_cache_indices(
    out_cache_loc: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens_tensor: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    extend_lens_tensor: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
    prefix_tensors: list[torch.Tensor],
    req_to_token_pool: ReqToTokenPool,
):
    pt = 0
    for i in range(req_pool_indices_cpu.shape[0]):
        req_idx = req_pool_indices_cpu[i].item()
        prefix_len = prefix_lens_cpu[i].item()
        seq_len = seq_lens_cpu[i].item()
        extend_len = extend_lens_cpu[i].item()

        req_to_token_pool.write(
            (req_idx, slice(0, prefix_len)),
            prefix_tensors[i],
        )
        req_to_token_pool.write(
            (req_idx, slice(prefix_len, seq_len)),
            out_cache_loc[pt : pt + extend_len],
        )
        pt += extend_len
