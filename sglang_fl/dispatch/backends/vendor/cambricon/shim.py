"""Cambricon early torch_mlu facade shim (activation-time).

neuware4.4.3's torch_mlu exposes only a partial CUDA-migration facade, so
sglang's stock code paths that assume a full CUDA alias break before any of
the cambricon vendor patches can run:

  * `pynccl_allocator` imports the torch.cuda.memory mempool symbols
    `_cuda_beginAllocateCurrentThreadToPool` / `_cuda_endAllocateToPool`,
    which torch_mlu does not provide → ImportError during sglang import.
  * `breakable_cuda_graph` does `isinstance(x, torch.Stream)`, but torch_mlu
    exposes `torch.Stream` as a wrapper (not a class) → annotation-style
    checks fail.

This module applies itself on import (module-body side effect), so importing
it from `activate_platform` — the earliest per-process hook, which fires in
the main process and every spawned scheduler/detokenizer child before the
quantization → fp8_kernel → deep_gemm_wrapper → pynccl_allocator and
dsa/utils → breakable_cuda_graph chains — puts the facade in place before
those symbols are first accessed. All injections are guarded (no-op when the
symbol already exists), so this is safe on neuware4.7.2 and later.
"""

import logging

logger = logging.getLogger(__name__)

_shim_applied = False


def _mlu_pool_noop(*args, **kwargs):
    return None


def _inject_mempool_symbols() -> None:
    import torch
    import torch_mlu.mlu.memory as mlu_memory

    for module in {getattr(torch.cuda, "memory", None), mlu_memory}:
        if module is None:
            continue
        for name in ("_cuda_beginAllocateCurrentThreadToPool", "_cuda_endAllocateToPool"):
            if not hasattr(module, name):
                setattr(module, name, _mlu_pool_noop)


def _rewrite_cuda_device(value):
    import torch

    if isinstance(value, str):
        if "cuda" in value:
            return value.replace("cuda", "mlu")
        if "CUDA" in value:
            return value.replace("CUDA", "MLU")
    elif isinstance(value, torch.device) and value.type == "cuda":
        index = f":{value.index}" if value.index is not None else ""
        return torch.device(f"mlu{index}")
    return value


def _restore_stream_class() -> None:
    import torch

    stream = torch.Stream
    if isinstance(stream, type):
        return
    orig = getattr(stream, "__wrapped__", None)
    if not isinstance(orig, type):
        logger.warning("cambricon shim: torch.Stream not restorable (%r)", stream)
        return

    class _MetaStream(type):
        def __instancecheck__(cls, instance):
            type_obj = type(instance)
            return (
                type_obj is cls
                or type_obj is orig
                or type_obj is torch.mlu.Stream
                or issubclass(type_obj, cls)
            )

        def __subclasscheck__(cls, subclass):
            if subclass is cls or subclass is orig:
                return True
            return type.__subclasscheck__(cls, subclass)

    class MLUStream(orig, metaclass=_MetaStream):
        base_class = orig

        def __new__(cls, *args, **kwargs):
            if args:
                args = tuple(_rewrite_cuda_device(a) for a in args)
            for key in ("device", "_device"):
                if key in kwargs:
                    kwargs[key] = _rewrite_cuda_device(kwargs[key])
            if not args and "device" not in kwargs and "_device" not in kwargs:
                kwargs["device"] = torch.device("mlu")
            return super().__new__(cls, *args, **kwargs)

    torch.Stream = MLUStream


def apply_early_shim() -> None:
    global _shim_applied
    if _shim_applied:
        return
    _shim_applied = True
    _inject_mempool_symbols()
    _restore_stream_class()
    logger.info("cambricon early shim applied")


apply_early_shim()
