#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Restore SGLang's native scalar-type fallback in the appfix5 CI image."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import stat
import tempfile


FINDER_MODULE = "_hcu_kernel_stub_finder"
SCALAR_MODULE = "sgl_kernel.scalar_type"
ORIGINAL = """    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(\".\")[0] not in _STUB_ROOTS:
            return None
        return importlib.util.spec_from_loader(fullname, _StubLoader())
"""
PATCHED = """    def find_spec(self, fullname, path=None, target=None):
        # scalar_type is optional in SGLang and has its own ImportError
        # fallback.  Pretending it exists changes scalar_types into a generic
        # stub function and breaks import-time attribute access (uint4b8).
        if fullname == \"sgl_kernel.scalar_type\":
            return None
        if fullname.split(\".\")[0] not in _STUB_ROOTS:
            return None
        return importlib.util.spec_from_loader(fullname, _StubLoader())
"""


def patch_source(source: str) -> tuple[str, bool]:
    """Return the precisely patched finder source and whether it changed."""
    if PATCHED in source:
        return source, False
    if source.count(ORIGINAL) != 1:
        raise RuntimeError(
            "HCU kernel-stub finder does not match the reviewed appfix5 source"
        )
    return source.replace(ORIGINAL, PATCHED, 1), True


def main() -> int:
    spec = importlib.util.find_spec(FINDER_MODULE)
    if spec is None or not spec.origin:
        raise RuntimeError(f"CI image does not provide {FINDER_MODULE}")
    destination = Path(spec.origin).resolve()
    if destination.suffix != ".py":
        raise RuntimeError(f"unexpected finder source path: {destination}")

    source = destination.read_text(encoding="utf-8")
    updated, changed = patch_source(source)
    if not changed:
        print(f"[hygon-ci] scalar fallback already enabled in {destination}")
        return 0

    mode = stat.S_IMODE(destination.stat().st_mode)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=destination.name + ".",
        delete=False,
    ) as output:
        output.write(updated)
        temporary = Path(output.name)
    os.chmod(temporary, mode)
    os.replace(temporary, destination)
    print(f"[hygon-ci] enabled SGLang scalar fallback in {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
