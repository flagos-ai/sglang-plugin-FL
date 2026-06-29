# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified structured logger for the ``patches`` subsystem.

Produces log lines with a consistent ``[TXDA_PATCH|module] action=... detail=...``
format, making it easy to query by module or action via grep/structured logging tools.

Usage::

    from sglang_fl.dispatch.backends.vendor.txda.patches._logger import patch_logger

    _log = patch_logger("device_support")

    _log.applied("added 'txda' to SUPPORTED_DEVICES (%s)", devices)
    _log.skipped("'txda' already in SUPPORTED_DEVICES")
    _log.failed("failed to patch SUPPORTED_DEVICES: %s", exc)

Query examples::

    grep 'TXDA_PATCH' logfile               # all patch logs
    grep 'TXDA_PATCH.*action=failed' logfile  # all failures
    grep 'TXDA_PATCH|dist_init' logfile       # specific module
"""

import logging


class PatchLogger:
    """Logger wrapper that emits structured ``[TXDA_PATCH|module]`` log lines.

    Each method corresponds to a standard action tag (``applied``, ``skipped``,
    ``failed``, ``info``, ``debug``, ``warning``), which becomes a fixed
    ``action=...`` field in the output.
    """

    def __init__(self, module: str) -> None:
        self._module = module
        self._logger = logging.getLogger(f"sglang_fl.dispatch.backends.vendor.txda.patches.{module}")

    def _format(self, action: str, msg: str) -> str:
        return f"[TXDA_PATCH|{self._module}] action={action} {msg}"

    def applied(self, msg: str, *args: object) -> None:
        self._logger.info(self._format("applied", msg), *args)

    def skipped(self, msg: str, *args: object) -> None:
        self._logger.info(self._format("skipped", msg), *args)

    def failed(self, msg: str, *args: object) -> None:
        self._logger.warning(self._format("failed", msg), *args)

    def info(self, msg: str, *args: object) -> None:
        self._logger.info(self._format("info", msg), *args)

    def debug(self, msg: str, *args: object) -> None:
        self._logger.debug(self._format("debug", msg), *args)

    def warning(self, msg: str, *args: object) -> None:
        self._logger.warning(self._format("warning", msg), *args)


def patch_logger(module: str) -> PatchLogger:
    """Create a :class:`PatchLogger` for the given module name.

    Args:
        module: Short module identifier (e.g. ``"device_support"``,
                ``"dist_init"``, ``"unquant"``, ``"patches"``).
    """
    return PatchLogger(module)
