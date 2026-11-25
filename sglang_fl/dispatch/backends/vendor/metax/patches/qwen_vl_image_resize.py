"""Resize 320x320 images before Qwen-VL preprocessing on MetaX.

Ported from the MetaX-tuned SGLang build (``multimodal/processors/qwen_vl.py``),
which resizes decoded 320x320 images to 288x288 before the HF processor creates
patches. Without it, concurrent VL requests on MetaX intermittently answer a
320x320 image with another request's answer (e.g. stop_sign -> "red").
Other sizes are left untouched.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import logging

import torch
import torchvision
from PIL import Image

logger = logging.getLogger(__name__)

_MODULE = "sglang.srt.multimodal.processors.qwen_vl"
_SRC_SIZE = 320
_DST_SIZE = 288
_PATCHED_MARKER = "__sglang_fl_metax_image_resize__"


def _resize_loaded_images(images):
    if not images:
        return images
    for i, img in enumerate(images):
        if isinstance(img, Image.Image):
            if img.size == (_SRC_SIZE, _SRC_SIZE):
                images[i] = img.convert("RGB").resize(
                    (_DST_SIZE, _DST_SIZE), Image.BILINEAR
                )
        elif isinstance(img, torch.Tensor):
            # GPU-decoded JPEG arrives as [C, H, W] uint8.
            if img.ndim == 3 and tuple(img.shape[-2:]) == (_SRC_SIZE, _SRC_SIZE):
                images[i] = torchvision.transforms.functional.resize(
                    img.unsqueeze(0), [_DST_SIZE, _DST_SIZE], antialias=True
                ).squeeze(0)
    return images


def patch_qwen_vl_image_resize() -> None:
    """Wrap QwenVLImageProcessor.load_mm_data to apply the 320x320 resize."""
    module = importlib.import_module(_MODULE)
    cls = module.QwenVLImageProcessor
    original = cls.load_mm_data
    if getattr(original, _PATCHED_MARKER, False):
        return

    if inspect.iscoroutinefunction(original):  # SGLang >= 0.5.16

        @functools.wraps(original)
        async def load_mm_data_metax(self, *args, **kwargs):
            output = await original(self, *args, **kwargs)
            _resize_loaded_images(getattr(output, "images", None))
            return output

    else:

        @functools.wraps(original)
        def load_mm_data_metax(self, *args, **kwargs):
            output = original(self, *args, **kwargs)
            _resize_loaded_images(getattr(output, "images", None))
            return output

    setattr(load_mm_data_metax, _PATCHED_MARKER, True)
    cls.load_mm_data = load_mm_data_metax
    logger.info("patched QwenVLImageProcessor.load_mm_data: 320x320 -> 288x288")
