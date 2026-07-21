"""Ascend/NPU patch for SGLang's Qwen-VL image/video preprocess.

The original module
``sglang.srt.hardware_backend.npu.modules.qwen_vl_processor`` fails to import on
the target transformers version because it imports
``group_images_by_shape``/``reorder_images`` from
``transformers.image_processing_utils_fast`` (moved to
``transformers.image_transforms``). Since the module can't be imported, we can't
monkey-patch its attributes the usual way.

Instead we install a lightweight shim module into ``sys.modules`` under the
original path. ``base_processor`` does a lazy
``from ...qwen_vl_processor import npu_apply_qwen_image_preprocess_patch`` at
request time, so it will pick up the shim, which lazily delegates to the fixed
implementation in ``_qwen_vl_processor_impl`` (deferring the transformers import
to first use, matching the original timing).
"""

from __future__ import annotations

import importlib
import logging
import sys
import types

logger = logging.getLogger(__name__)

_SGLANG_QWEN_MODULE = "sglang.srt.hardware_backend.npu.modules.qwen_vl_processor"
_IMPL_MODULE = (
    "sglang_fl.dispatch.backends.vendor.ascend.patches._qwen_vl_processor_impl"
)


def patch_qwen_vl_processor() -> None:
    if _SGLANG_QWEN_MODULE in sys.modules:
        # The real (broken) module is already imported — don't clobber it.
        logger.warning(
            "Ascend qwen_vl_processor patch skipped: %s already imported",
            _SGLANG_QWEN_MODULE,
        )
        return

    shim = types.ModuleType(_SGLANG_QWEN_MODULE)
    shim.__doc__ = "sglang_fl Ascend shim for qwen_vl_processor (lazy)."

    def npu_apply_qwen_image_preprocess_patch():
        impl = importlib.import_module(_IMPL_MODULE)
        return impl.npu_apply_qwen_image_preprocess_patch()

    def __getattr__(name):
        # Lazily expose any other public symbol from the fixed implementation.
        impl = importlib.import_module(_IMPL_MODULE)
        return getattr(impl, name)

    shim.npu_apply_qwen_image_preprocess_patch = npu_apply_qwen_image_preprocess_patch
    shim.__getattr__ = __getattr__

    sys.modules[_SGLANG_QWEN_MODULE] = shim
    logger.info("Ascend qwen_vl_processor shim installed into sys.modules")
