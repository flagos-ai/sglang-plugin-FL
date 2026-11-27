# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Adapt SGLang's INT8 scaled-mm contract to FlagGems' public output API.

The historical ``sm90`` API name denotes the generic Triton INT8 implementation
in FlagGems 97dbf2fd19cb9d81d5115399f7e765be5f7a7ec6. No CUDA/CUTLASS
binary is called. Keep this implementation in the generic FlagOS backend;
it is not a vendor-native MUSA implementation.
"""

import inspect


def is_int8_scaled_mm_available():
    from sglang_fl.utils import get_device_info

    info = get_device_info()
    if info is None or info.vendor_name != "mthreads":
        return False
    try:
        from flag_gems.fused.cutlass_scaled_mm import cutlass_scaled_mm_sm90
    except ImportError:
        return False
    return all(
        name in inspect.signature(cutlass_scaled_mm_sm90).parameters
        for name in ("c", "a", "b", "a_scale", "b_scale", "bias")
    )


def int8_scaled_mm_flagos(a, b, a_scale, b_scale, *, out_dtype, bias=None):
    """A[M,K] @ B[K,N], token/channel FP32 scales, output[M,N].

    Ordinary output allocation is local to the call. There is no cross-request
    state, persistent workspace or execution-time backend retry.
    """
    import torch
    from flag_gems.fused.cutlass_scaled_mm import cutlass_scaled_mm_sm90

    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("INT8 scaled-mm requires compatible rank-two matrices")
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise TypeError("INT8 scaled-mm requires INT8 matrices")
    if a.device != b.device or a.device.type != "musa":
        raise ValueError("INT8 scaled-mm requires matrices on the same MUSA device")
    if out_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("INT8 scaled-mm output must be FP16 or BF16")
    m, n = a.shape[0], b.shape[1]
    for scale, size in ((a_scale, m), (b_scale, n)):
        if scale.dtype != torch.float32 or scale.device != a.device:
            raise TypeError("INT8 scaled-mm scales must be FP32 on the input device")
        if scale.numel() != size:
            raise ValueError("INT8 scaled-mm requires token/channel scales")
    if bias is not None:
        raise NotImplementedError("MUSA INT8 scaled-mm does not yet support bias")
    if a.stride(1) != 1 or b.stride(0) != 1:
        raise ValueError("INT8 scaled-mm requires row-major A and column-major B")
    output = torch.empty((m, n), device=a.device, dtype=out_dtype)
    if m and n:
        cutlass_scaled_mm_sm90(
            output,
            a,
            b,
            a_scale.reshape(m).contiguous(),
            b_scale.reshape(n).contiguous(),
            bias,
        )
    return output
