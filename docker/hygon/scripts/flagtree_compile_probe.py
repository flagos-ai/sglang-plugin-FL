#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""GPU-less probe of FlagTree's REAL HCU compile path (post-build gate).

Runs INSIDE the finished image (docker/hygon/scripts/verify_image.sh phase 4).
It does not call clang by hand: it drives FlagTree's own
``HIPBackend.make_amdgcn`` and ``HIPBackend.make_hsaco`` on the
OCML-referencing probe IR - the exact code path every Triton/FlagGems kernel
takes on HCU - with FlagTree's default configuration (no
``TRITON_HIP_CLANG_PATH`` override, no image-injected wrapper).

What it proves on a GPU-less host:
  * FlagTree selects the Runtime's own clang (``/opt/dtk/aillvm/bin/clang-18``
    via ``ROCM_PATH``);
  * the four device libraries FlagTree links into the IR exist;
  * clang's device-library discovery succeeds natively, so IR that still
    declares ``__ocml_*`` calls lowers, links and assembles to a code object.

It replaces the hand-written two-step clang invocation the acceptance used
before: that invocation never carried FlagTree's real argv and therefore
could not tell whether the wrapper the reviewer rejected was still needed.
This is a compile-path proof only - kernels executing on the device are the
hardware matrix's job.

Usage: flagtree_compile_probe.py --probe /scripts/hcu-device-lib-check.ll
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

EXPECTED_CLANG = Path("/opt/dtk/aillvm/bin/clang-18")
EXPECTED_ARCH = "gfx936"
WRAPPER_OVERRIDE = "TRITON_HIP_CLANG_PATH"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    args = parser.parse_args()

    assert WRAPPER_OVERRIDE not in os.environ, (
        f"{WRAPPER_OVERRIDE} is set: an image-injected clang override remains"
    )
    from triton.backends.hcu.compiler_hcu import HIPBackend, HIPOptions

    clang = HIPBackend.path_to_rocm_clang()
    assert clang == EXPECTED_CLANG, (
        f"FlagTree selected {clang}, expected {EXPECTED_CLANG}"
    )
    options = HIPOptions(arch=EXPECTED_ARCH)
    extern_libs = dict(options.extern_libs)
    missing = [path for path in extern_libs.values() if not Path(path).is_file()]
    assert not missing, f"FlagTree device libraries missing: {missing}"
    clang_args = HIPBackend._get_clang_args({}, options)

    metadata: dict = {}
    assembly = HIPBackend.make_amdgcn(args.probe.read_text(), metadata, options)
    code_object = HIPBackend.make_hsaco(assembly, metadata, options)
    assert code_object, "FlagTree produced an empty code object"

    # does a device-library path really enter clang's cc1 command line?
    import subprocess

    inspect = subprocess.run(
        [str(clang), "-###", *clang_args, str(args.probe), "-S", "-o", "/dev/null"],
        capture_output=True,
        text=True,
        check=True,
    )
    bitcode_dirs = sorted(
        {
            str(Path(token.rstrip(",")).parent)
            for token in inspect.stderr.replace('"', " ").split()
            if token.endswith(".bc")
        }
    )
    assert bitcode_dirs, "no ROCm device library enters clang's cc1 command line"

    print(
        "FlagTree HCU compile path: "
        + json.dumps(
            {
                "clang": str(clang),
                "rocm": HIPBackend.path_to_rocm(),
                "device_lib_dir": bitcode_dirs[0],
                "code_object_bytes": len(code_object),
                "code_object_sha256": hashlib.sha256(code_object).hexdigest(),
                "clang_args": clang_args,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as failure:
        print(f"FATAL: {failure}", file=sys.stderr)
        raise SystemExit(1) from failure
