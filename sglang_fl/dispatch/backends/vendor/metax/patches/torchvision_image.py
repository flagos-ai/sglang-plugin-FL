"""Patch torchvision image decode paths that are unsafe on MetaX."""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import logging
import sys
import types
from typing import Any, Callable

logger = logging.getLogger(__name__)

_TORCHVISION_IMAGE_MODULE = "torchvision.io.image"
_TORCHVISION_IO_MODULE = "torchvision.io"
_SGLANG_COMMON_MODULE = "sglang.srt.utils.common"
_FINDER_MARKER = "__sglang_fl_metax_torchvision_image_finder__"
_PATCHED_MARKER = "__sglang_fl_metax_decode_jpeg_cpu_patch__"
_LOAD_IMAGE_PATCHED_MARKER = "__sglang_fl_metax_load_image_cpu_patch__"
_ORIGINAL_ATTR = "__sglang_fl_metax_original_decode_jpeg__"


def _is_accelerator_device(device: Any) -> bool:
    if device is None:
        return False
    text = str(device).strip().lower()
    return text == "maca" or text.startswith("cuda")


def _wrap_decode_jpeg(fn: Callable) -> Callable:
    if getattr(fn, _PATCHED_MARKER, False):
        return fn

    @functools.wraps(fn)
    def decode_jpeg_cpu_on_metax(*args, **kwargs):
        if "device" in kwargs:
            if _is_accelerator_device(kwargs["device"]):
                kwargs = dict(kwargs)
                kwargs["device"] = "cpu"
        elif len(args) >= 3 and _is_accelerator_device(args[2]):
            args = list(args)
            args[2] = "cpu"
            args = tuple(args)
        return fn(*args, **kwargs)

    setattr(decode_jpeg_cpu_on_metax, _PATCHED_MARKER, True)
    setattr(decode_jpeg_cpu_on_metax, _ORIGINAL_ATTR, fn)
    return decode_jpeg_cpu_on_metax


def _wrap_load_image(fn: Callable) -> Callable:
    if getattr(fn, _LOAD_IMAGE_PATCHED_MARKER, False):
        return fn

    @functools.wraps(fn)
    def load_image_cpu_decode_on_metax(*args, **kwargs):
        if len(args) >= 3:
            args = list(args)
            args[2] = False
            args = tuple(args)
        else:
            kwargs = dict(kwargs)
            kwargs["gpu_image_decode"] = False
        return fn(*args, **kwargs)

    setattr(load_image_cpu_decode_on_metax, _LOAD_IMAGE_PATCHED_MARKER, True)
    setattr(load_image_cpu_decode_on_metax, _ORIGINAL_ATTR, fn)
    return load_image_cpu_decode_on_metax


def _patch_module_decode_jpeg(module: types.ModuleType) -> None:
    decode_jpeg = getattr(module, "decode_jpeg", None)
    if decode_jpeg is None:
        return
    if getattr(decode_jpeg, _PATCHED_MARKER, False):
        return
    setattr(module, "decode_jpeg", _wrap_decode_jpeg(decode_jpeg))
    logger.info("patched %s.decode_jpeg to force CPU JPEG decode on MetaX", module.__name__)


def _patch_module_load_image(module: types.ModuleType) -> None:
    load_image = getattr(module, "_load_image", None)
    if load_image is None:
        return
    if getattr(load_image, _LOAD_IMAGE_PATCHED_MARKER, False):
        return
    setattr(module, "_load_image", _wrap_load_image(load_image))
    logger.info(
        "patched %s._load_image to disable GPU JPEG decode on MetaX",
        module.__name__,
    )


def _patch_module(module: types.ModuleType) -> None:
    _patch_module_decode_jpeg(module)
    if module.__name__ == _SGLANG_COMMON_MODULE:
        _patch_module_load_image(module)


def _patch_loaded_modules() -> None:
    for module_name in (
        _TORCHVISION_IMAGE_MODULE,
        _TORCHVISION_IO_MODULE,
        _SGLANG_COMMON_MODULE,
    ):
        module = sys.modules.get(module_name)
        if isinstance(module, types.ModuleType):
            _patch_module(module)


class _PatchTorchvisionImageLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader):
        self._wrapped_loader = wrapped_loader

    def create_module(self, spec):
        create_module = getattr(self._wrapped_loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module):
        self._wrapped_loader.exec_module(module)
        _patch_module(module)
        _patch_loaded_modules()


class _PatchTorchvisionImageFinder(importlib.abc.MetaPathFinder):
    __sglang_fl_metax_torchvision_image_finder__ = True

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in {
            _TORCHVISION_IMAGE_MODULE,
            _TORCHVISION_IO_MODULE,
            _SGLANG_COMMON_MODULE,
        }:
            return None

        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        if isinstance(spec.loader, _PatchTorchvisionImageLoader):
            return spec
        if not hasattr(spec.loader, "exec_module"):
            return None

        spec.loader = _PatchTorchvisionImageLoader(spec.loader)
        return spec


def patch_torchvision_decode_jpeg_cuda_to_cpu() -> None:
    """Avoid MetaX VPU/JPEG native crashes from torchvision CUDA JPEG decode."""
    _patch_loaded_modules()

    for finder in sys.meta_path:
        if getattr(finder, _FINDER_MARKER, False):
            return
    sys.meta_path.insert(0, _PatchTorchvisionImageFinder())
    logger.info("installed MetaX torchvision decode_jpeg import hook")
