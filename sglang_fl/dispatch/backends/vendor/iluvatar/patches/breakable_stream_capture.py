# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Iluvatar/CoreX compatibility for capture queries on a specified stream."""

import importlib

_TARGET_MODULE = (
    "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph."
    "breakable_cuda_graph"
)
_applied = False
_original_is_stream_capturing = None


def _is_stream_capturing_corex(stream):
    """Query the requested stream without changing the caller's stream."""
    import torch

    with torch.cuda.stream(stream):
        return torch.cuda.is_current_stream_capturing()


def patch_breakable_stream_capture() -> bool:
    """Use CoreX capture queries within the requested stream context."""
    global _applied, _original_is_stream_capturing
    from sglang_fl.utils import get_device_info

    info = get_device_info()
    if info is None or info.vendor_name != "iluvatar":
        return False
    if _applied:
        return True

    import torch

    if not hasattr(torch, "corex"):
        raise RuntimeError("Iluvatar stream-capture compatibility requires CoreX Torch")
    cuda = getattr(torch, "cuda", None)
    if not callable(getattr(cuda, "stream", None)) or not callable(
        getattr(cuda, "is_current_stream_capturing", None)
    ):
        raise RuntimeError("CoreX stream context and capture-query APIs are required")

    module = importlib.import_module(_TARGET_MODULE)
    original = getattr(module, "_is_stream_capturing", None)
    if not callable(original):
        raise RuntimeError(
            "Actual breakable CUDA graph capture-query binding is missing"
        )

    _original_is_stream_capturing = original
    module._is_stream_capturing = _is_stream_capturing_corex
    _applied = True
    return True
