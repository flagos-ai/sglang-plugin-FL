"""Extend sglang's SUPPORTED_DEVICES with this vendor's device_type."""

import logging

logger = logging.getLogger(__name__)


def patch_supported_devices():
    from sglang.srt.configs import device_config as dc

    # Replace "template" with your torch device_type (e.g. "musa", "npu", "gcu").
    if "template" not in dc.SUPPORTED_DEVICES:
        dc.SUPPORTED_DEVICES = [*dc.SUPPORTED_DEVICES, "template"]
        logger.info("patched SUPPORTED_DEVICES += [template]")
