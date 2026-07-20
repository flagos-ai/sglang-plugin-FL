import logging

logger = logging.getLogger(__name__)


def patch_supported_devices():
    from sglang.srt.configs import device_config as dc

    if "gcu" not in dc.SUPPORTED_DEVICES:
        dc.SUPPORTED_DEVICES = [*dc.SUPPORTED_DEVICES, "gcu"]
        logger.info("patched SUPPORTED_DEVICES += [gcu]")