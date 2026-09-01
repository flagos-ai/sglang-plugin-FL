# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/utils.py
# Copied from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/utils.py
# -*- coding: utf-8 -*-
#
# Trimmed vendor copy: only keeps `input_guard`, which is depended on by
# `fused_sigmoid_gating_recurrent.py`. The full chunk_gated_delta_rule pipeline
# was removed — that op is now expected to fall back through the plugin
# dispatch system (vendor → flagos → reference).

import contextlib
import functools
from typing import Callable

import torch


def custom_device_ctx(index: int):
    return torch.npu.device(index)


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (
            i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args
        )
        contiguous_kwargs = {
            k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
            for k, v in kwargs.items()
        }

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = custom_device_ctx(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper
