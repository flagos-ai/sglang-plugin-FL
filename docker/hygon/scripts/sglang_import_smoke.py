#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Post-install gate 4: sglang import smoke check.

A GPU-less builder cannot load device libraries (libgalaxyhip / libcuda /
torch device backends), so those failures are recorded and tolerated; any
other failure - notably a missing pure-Python dependency of the closure -
fails the build. With the loader path baked into the image a GPU-less
builder can now load the DTK libraries and may abort (SIGABRT, rc=134)
inside HIP device init before Python can classify the failure, so the
verdict comes from log sentinels, not the exit code.
"""

import importlib.metadata as md
import traceback

try:
    import sglang

    print("SMOKE-PASS sglang", md.version("sglang"), "imported from", sglang.__file__)
except BaseException:  # noqa: BLE001 - classify below
    text = traceback.format_exc()
    print(text)
    tolerated = (
        "libgalaxyhip" in text,
        "libcuda.so" in text,
        "Failed to load the backend extension" in text,
        "cannot open shared object file" in text,
    )
    if not any(tolerated):
        raise SystemExit(
            "sglang import failed for reasons other than missing device libraries"
        )
    print(
        "SMOKE-TOLERATED sglang import blocked by device libraries only "
        "(expected on a GPU-less builder)"
    )
