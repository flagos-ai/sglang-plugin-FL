# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Non-owning CUDA tensor metadata for the explicitly installed CoreX addon."""


def weak_ref_tensor(tensor):
    """Make a tensor with independent, non-owning storage over the same bytes.

    This follows CoreX Torch's CUDAGraphNode reconstruction path. The original
    Tensor and Storage are not retained: callers must keep the underlying
    allocation alive, for example through the captured graph's memory pool.
    No copy, accelerator allocation, or Python weakref proxy is involved.
    """
    import torch

    # Match sgl_kernel.memory's non-Tensor pass-through contract.
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if not hasattr(torch, "corex"):
        raise RuntimeError("sgl_kernel.weak_ref_tensor requires Iluvatar CoreX Torch")
    if not tensor.is_cuda or tensor.layout != torch.strided or tensor.is_quantized:
        raise RuntimeError("sgl_kernel.weak_ref_tensor requires a strided CUDA tensor")
    if tensor.is_conj() or tensor.is_neg():
        raise RuntimeError(
            "sgl_kernel.weak_ref_tensor cannot reconstruct lazy view bits"
        )

    make_storage = getattr(torch._C, "_construct_storage_from_data_pointer", None)
    make_tensor = getattr(
        torch._C, "_construct_CUDA_Tensor_From_Storage_And_Metadata", None
    )
    if not callable(make_storage) or not callable(make_tensor):
        raise RuntimeError(
            "CoreX Torch lacks non-owning CUDA tensor reconstruction APIs"
        )

    # Use the storage base and the actual offset together. Using data_ptr()
    # with this offset would apply it twice to sliced/non-contiguous tensors.
    source_storage = tensor.untyped_storage()
    metadata = {
        "nbytes": source_storage.nbytes(),
        "data_ptr": source_storage.data_ptr(),
        "size": tuple(tensor.size()),
        "stride": tuple(tensor.stride()),
        "dtype": tensor.dtype,
        "device": tensor.device,
        "storage_offset": tensor.storage_offset(),
    }
    storage = make_storage(metadata["data_ptr"], metadata["device"], metadata["nbytes"])
    return make_tensor(metadata, storage)
