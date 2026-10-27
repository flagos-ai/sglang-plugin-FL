# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Route Ascend BF16 dense linear through the configured ATen mm dispatch."""

from __future__ import annotations

from functools import wraps
import inspect

import torch


_PATCH_MARKER = "_sglang_fl_npu_bf16_unquant_mm"
_APPLY_PARAMETERS = ("self", "layer", "x", "bias")


def patch_unquantized_linear_mm() -> None:
    """Expose the configured mm implementation for NPU BF16 matrix inputs.

    TorchNPU's F.linear can bypass the registered ATen mm implementation. Use
    torch.mm for the validated two-dimensional, unbiased BF16 case so the
    existing operator policy controls the GEMM. This also respects the normal
    native dispatch when FlagGems is disabled; no backend is forced here.
    """
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    original = UnquantizedLinearMethod.apply
    if getattr(original, _PATCH_MARKER, False):
        return
    signature = inspect.signature(original)
    parameters = tuple(signature.parameters.values())
    if (
        tuple(signature.parameters) != _APPLY_PARAMETERS
        or any(p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD for p in parameters)
        or any(p.default != inspect.Parameter.empty for p in parameters[:3])
        or parameters[-1].default is not None
    ):
        raise RuntimeError(
            "Unsupported SGLang interface for the Ascend unquantized linear patch: "
            f"{original.__qualname__}{signature}."
        )

    @wraps(original)
    def apply(self, layer, x, bias=None):
        # FlagGems rejects a zero-sized grid; keep empty matrices on the native path.
        if (
            x.device.type != "npu"
            or x.ndim != 2
            or x.shape[0] == 0
            or bias is not None
            or x.dtype != torch.bfloat16
        ):
            return original(self, layer, x, bias)
        weight = layer.weight
        if weight.dtype != torch.bfloat16:
            return original(self, layer, x, bias)
        if (
            weight.ndim != 2
            or weight.device != x.device
            or x.shape[-1] != weight.shape[-1]
        ):
            raise RuntimeError("Ascend unquantized mm shape/device contract changed")

        output = torch.mm(x, weight.t())
        if (
            output.shape != (x.shape[0], weight.shape[0])
            or output.dtype != x.dtype
            or output.device != x.device
        ):
            raise RuntimeError("Ascend unquantized mm output contract changed")
        return output

    setattr(apply, _PATCH_MARKER, True)
    UnquantizedLinearMethod.apply = apply
