# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Identify the affected FlagTree stack on Ascend."""

import importlib.util
import os
from importlib import metadata
from pathlib import Path

# FlagTree builds carrying the known CANN 9.2.1 defects the workarounds below
# compensate for (FlagTree #1668 indirect-pointer load, FlagGems #7207/#6802
# gelu/lift_fresh). The operations Runtime 2.3.0 ships 0.7.1a1+ascend3.5; the
# workarounds stay active on it until the hardware matrix proves the defects
# are gone, because both workarounds are no-ops when their trigger conditions
# do not fire.
_AFFECTED_FLAGTREE_VERSIONS = frozenset(
    (
        "0.7.0+ascend3.5",
        "0.7.1a1+ascend3.5",
    )
)


def uses_flagtree_cann921() -> bool:
    try:
        if metadata.version("sglang") != "0.5.18":
            return False
        flagtree = metadata.distribution("flagtree")
    except metadata.PackageNotFoundError:
        return False
    if flagtree.version not in _AFFECTED_FLAGTREE_VERSIONS:
        return False

    triton = importlib.util.find_spec("triton")
    owned_triton = Path(flagtree.locate_file("triton/__init__.py")).resolve()
    if (
        triton is None
        or triton.origin is None
        or Path(triton.origin).resolve() != owned_triton
    ):
        return False

    toolkit = Path(
        os.environ.get("ASCEND_HOME_PATH")
        or os.environ.get("ASCEND_HOME")
        or os.environ.get("ASCEND_TOOLKIT_HOME")
        or "/usr/local/Ascend/ascend-toolkit/latest"
    )
    for directory in ("aarch64-linux", "arm64-linux", "."):
        for filename in ("ascend_all_cann_install.info", "ascend_toolkit_install.info"):
            info = toolkit / directory / filename
            try:
                lines = info.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line in lines:
                name, separator, value = line.partition("=")
                if separator and name.strip().lower() == "version":
                    return value.strip().strip("\"'") == "9.2.1"
    return False
