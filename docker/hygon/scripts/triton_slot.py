#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""The Triton compiler slots of the Hygon operations Runtime.

The Runtime ships the Triton compiler as module TREES under fixed roots,
not as something the image resolver installs; Runtime's Bash initialization
selects the production tree through ``PYTHONPATH``:

  /opt/flagtree  the tree Runtime Bash selects. Production default.
  /opt/triton    a second tree the Runtime ships for A/B experiments.

What each tree contains is decided by the Runtime owner and changes between
Runtime rebuilds, so this module only knows the roots and how to OBSERVE a
tree. The expected contents are never written here: the build records what
the resolved base digest ships (``collect_runtime_reference.py``) and the
acceptance compares the finished image against that record.
"""

from __future__ import annotations

import re
from pathlib import Path

FLAGTREE_SLOT = "/opt/flagtree"
VENDOR_SLOT = "/opt/triton"
PRODUCTION_SLOT = FLAGTREE_SLOT
SLOT_ROOTS = (FLAGTREE_SLOT, VENDOR_SLOT)

_VERSION_RE = re.compile(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", re.MULTILINE)


def slot_exists(slot: str) -> bool:
    return (Path(slot) / "triton" / "__init__.py").is_file()


def inspect_slot(slot: str) -> dict:
    """Observed facts of a slot tree (read from disk; nothing is imported)."""
    base = Path(slot)
    init = base / "triton" / "__init__.py"
    match = _VERSION_RE.search(init.read_text()) if init.is_file() else None
    backends = base / "triton" / "backends"
    return {
        "module_version": match.group(1) if match else None,
        "triton_dist_info": sorted(p.name for p in base.glob("triton-*.dist-info")),
        "backends": sorted(
            p.name
            for p in backends.iterdir()
            if p.is_dir() and not p.name.startswith("_")
        )
        if backends.is_dir()
        else [],
    }
