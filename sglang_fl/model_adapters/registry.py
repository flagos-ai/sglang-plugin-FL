"""Register FlagOS model implementations without modifying SGLang sources."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_registered = False


def _register_transformers_configs(vendor: str) -> None:
    from sglang.srt.utils.hf_transformers.common import _CONFIG_REGISTRY, AutoConfig

    if vendor == "ascend":
        from sglang_fl.models.qwen4_exp_impl.config import (
            Qwen4ExpConfig,
            Qwen4ExpTextConfig,
        )

        configs = {
            Qwen4ExpConfig.model_type: Qwen4ExpConfig,
            Qwen4ExpTextConfig.model_type: Qwen4ExpTextConfig,
        }
    elif vendor == "mthreads":
        from sglang_fl.models.xingchen4_impl.config import (
            Xing4_0Config,
            XingChen4Config,
        )

        configs = {
            XingChen4Config.model_type: XingChen4Config,
            Xing4_0Config.model_type: Xing4_0Config,
        }
    else:
        return

    _CONFIG_REGISTRY.update(configs)
    for model_type, config_cls in configs.items():
        try:
            AutoConfig.register(model_type, config_cls)
        except ValueError as exc:
            message = str(exc).lower()
            if "already registered" not in message and "already used" not in message:
                raise


def _detect_vendor() -> str:
    override = os.environ.get("SGLANG_FL_MODEL_VENDOR", "").strip().lower()
    if override:
        return override
    try:
        try:
            from flag_gems.runtime.backend.device import DeviceDetector
        except ImportError:
            from flag_gems.runtime.backend.device_finder import DeviceDetector
        return str(DeviceDetector().vendor_name).lower()
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.warning("Model adapter vendor detection failed: %s", exc)
        return ""


def _register_sglang_models(vendor: str) -> None:
    from sglang.srt.models.registry import ModelRegistry

    packages = {
        "ascend": "sglang_fl.models.ascend_registry",
        "mthreads": "sglang_fl.models.mthreads_registry",
    }
    package = packages.get(vendor)
    if package is not None:
        # The package contains only the models valid for this accelerator. This
        # avoids importing MUSA kernels in an Ascend process (and vice versa).
        ModelRegistry.register(package, overwrite=True, strict=True)


def register_model_adapters() -> None:
    """Install config/model registrations and compatibility hooks once."""
    global _registered
    if _registered:
        return

    vendor = _detect_vendor()
    _register_transformers_configs(vendor)
    _register_sglang_models(vendor)

    if vendor == "ascend":
        from .qwen4_exp import register_qwen4_exp_hooks

        register_qwen4_exp_hooks()
    elif vendor == "mthreads":
        from .xingchen4 import register_xingchen4_hooks

        register_xingchen4_hooks()
    _registered = True
    logger.info("Registered native model adapters for vendor=%s", vendor or "unknown")
