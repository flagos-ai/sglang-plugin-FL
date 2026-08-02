import torch
import triton
import triton.language as tl


@triton.jit
def _gemma_rmsnorm_kernel(
    X_ptr,
    W_ptr,
    Out_ptr,
    stride_x_row,
    stride_out_row,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_row_ptr = X_ptr + row_idx * stride_x_row
    out_row_ptr = Out_ptr + row_idx * stride_out_row

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / N
    rrms = tl.rsqrt(mean_sq + eps)

    out = x * rrms * (1.0 + w)

    tl.store(out_row_ptr + cols, out.to(tl.load(x_row_ptr + cols, mask=mask).dtype), mask=mask)


@triton.jit
def _gemma_fused_add_rmsnorm_kernel(
    X_ptr,
    Residual_ptr,
    W_ptr,
    Out_ptr,
    ResidualOut_ptr,
    stride_x_row,
    stride_res_row,
    stride_out_row,
    stride_resout_row,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_row_ptr = X_ptr + row_idx * stride_x_row
    res_row_ptr = Residual_ptr + row_idx * stride_res_row
    out_row_ptr = Out_ptr + row_idx * stride_out_row
    resout_row_ptr = ResidualOut_ptr + row_idx * stride_resout_row

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(res_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    hidden = x + residual

    orig_dtype = tl.load(x_row_ptr + cols, mask=mask).dtype
    tl.store(resout_row_ptr + cols, hidden.to(orig_dtype), mask=mask)

    mean_sq = tl.sum(hidden * hidden, axis=0) / N
    rrms = tl.rsqrt(mean_sq + eps)

    out = hidden * rrms * (1.0 + w)

    tl.store(out_row_ptr + cols, out.to(orig_dtype), mask=mask)


def _next_power_of_2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.is_contiguous()
    orig_shape = x.shape
    if x.dim() != 2:
        x = x.reshape(-1, orig_shape[-1])

    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = _next_power_of_2(N)

    _gemma_rmsnorm_kernel[(M,)](
        x, weight, out,
        x.stride(0), out.stride(0),
        N, eps,
        BLOCK_N=BLOCK_N,
    )

    if len(orig_shape) != 2:
        out = out.reshape(orig_shape)
    return out


def gemma_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous() and residual.is_contiguous()
    assert x.shape == residual.shape

    orig_shape = x.shape
    if x.dim() != 2:
        x = x.reshape(-1, orig_shape[-1])
        residual = residual.reshape(-1, orig_shape[-1])

    M, N = x.shape
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    BLOCK_N = _next_power_of_2(N)

    _gemma_fused_add_rmsnorm_kernel[(M,)](
        x, residual, weight, out, residual_out,
        x.stride(0), residual.stride(0), out.stride(0), residual_out.stride(0),
        N, eps,
        BLOCK_N=BLOCK_N,
    )

    if len(orig_shape) != 2:
        out = out.reshape(orig_shape)
        residual_out = residual_out.reshape(orig_shape)
    return out, residual_out


# def gemma_rms_norm_op(module, x, residual=None):
#     if residual is not None:
#         return gemma_fused_add_rmsnorm(x, residual, module.weight.data, module.variance_epsilon)
#     return gemma_rmsnorm(x, module.weight.data, module.variance_epsilon)


# def gemma_rms_norm_func(x, weight, eps=1e-6, residual=None):
#     if residual is not None:
#         return gemma_fused_add_rmsnorm(x, residual, weight, eps)
#     return gemma_rmsnorm(x, weight, eps)
def gemma_rms_norm(x, weight, eps=1e-6, residual=None):
    if residual is not None:
        return gemma_fused_add_rmsnorm(x, residual, weight, eps)
    return gemma_rmsnorm(x, weight, eps)
