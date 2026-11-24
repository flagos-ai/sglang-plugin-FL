#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""The two Triton compiler slots of the Hygon operations Runtime, as contracts.

The Runtime ships the Triton compiler as module TREES, not as something the
image resolver installs; the image-level ``PYTHONPATH`` selects which tree
every entrypoint imports:

  /opt/flagtree  FlagTree 0.7.0+hcu3.6: ``triton`` module 3.6.0, HCU backend,
                 no ``triton`` dist-info, so ``pip check`` does not see a
                 triton distribution at all. Production default.
  /opt/triton    DTK vendor triton 3.5.1 WITH its own dist-info, so
                 ``pip check`` and importlib.metadata DO see a triton
                 distribution (reviewer r4627164234).

The acceptance is parametrized by the slot an image selects
(``--selected-slot``; the production image selects /opt/flagtree); every
slot is checked against ITS OWN contract below, never against a loosened
subset both happen to satisfy. The facts were observed on
flagos-runtime-hygon-dtk26.04 digest ``acccfccb...``; a Runtime that
changes a slot must change this table as deliberately as the other
``EXPECTED_*`` constants.
"""

from __future__ import annotations

import re
from pathlib import Path

FLAGTREE_SLOT = "/opt/flagtree"
VENDOR_SLOT = "/opt/triton"
PRODUCTION_SLOT = FLAGTREE_SLOT

SLOT_CONTRACTS: dict[str, dict] = {
    FLAGTREE_SLOT: {
        "module_version": "3.6.0",
        "triton_dist_info": None,
        "backends": ("hcu",),
        "pip_check_sees_triton": False,
    },
    VENDOR_SLOT: {
        "module_version": "3.5.1",
        "triton_dist_info": "triton-3.5.1+das.opt1.dtk2604.torch290.dist-info",
        "backends": ("amd", "nvidia"),
        "pip_check_sees_triton": True,
    },
}

_VERSION_RE = re.compile(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", re.MULTILINE)


def contract_of(slot: str) -> dict:
    try:
        return SLOT_CONTRACTS[slot]
    except KeyError:
        known = ", ".join(sorted(SLOT_CONTRACTS))
        raise AssertionError(f"unknown triton slot {slot!r} (known: {known})") from None


def inspect_slot(root: str) -> dict:
    """Observed facts of a slot tree (read from disk; nothing is imported)."""
    base = Path(root)
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


def assert_slot_root(slot: str) -> dict:
    """The slot tree must still match its contract (it is Runtime-owned)."""
    expected = contract_of(slot)
    observed = inspect_slot(slot)
    expected_dist_info = (
        [expected["triton_dist_info"]] if expected["triton_dist_info"] else []
    )
    checks = {
        "module_version": observed["module_version"] == expected["module_version"],
        "triton_dist_info": observed["triton_dist_info"] == expected_dist_info,
        "backends": tuple(observed["backends"]) == tuple(expected["backends"]),
    }
    failed = [name for name, ok in checks.items() if not ok]
    assert not failed, (
        f"triton slot {slot} no longer matches its contract "
        f"({failed}): observed {observed}, expected {expected}"
    )
    return observed


def dist_info_version(dist_info: str | None) -> str | None:
    """``triton-3.5.1+das...dist-info`` -> ``3.5.1+das...``."""
    if dist_info is None:
        return None
    assert dist_info.startswith("triton-") and dist_info.endswith(".dist-info")
    return dist_info[len("triton-") : -len(".dist-info")]
