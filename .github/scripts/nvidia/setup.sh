#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Install sglang-plugin-FL and test dependencies on NVIDIA CUDA.
set -euo pipefail
git config --global --add safe.directory "$(pwd)"
echo "=== Minimal fast-path: swap triton only ==="
# TEMPORARY: the only functional change from the full-closure image is the
# triton version — 3.7.1 (from the standard closure) crashes kernel launches
# (SystemError in triton/backends/nvidia/driver.py). FlagTree provides 3.6.0
# at /opt/flagtree; installing the same version as a pip wheel prevents the
# unit-test sys.modules contamination while matching FlagTree's ABI.
# Everything else (CUDA toolchain, kernel stacks, all packages) stays.
python3 -m pip install --no-deps triton==3.6.0
echo "triton swapped: 3.7.1 -> 3.6.0 (FlagTree-matching version)"
echo "=== Installing test runner (ephemeral) ==="
# The operations-runtime CI image intentionally ships no test packages; the
# pinned runner installs into the ephemeral job container only, at the same
# versions the previous image baked in.
python3 -m pip install --break-system-packages \
    pytest==9.1.1 pytest-timeout==2.4.0
echo "=== Installing sglang-plugin-FL (CUDA) ==="
# The operations runtime's Python is externally managed (PEP 668). This is an
# ephemeral CI container, so install only the editable plugin and explicitly
# preserve every pinned dependency.
python3 -m pip install \
    --break-system-packages --no-deps --no-build-isolation -e .
python3 .github/scripts/nvidia/verify_environment.py \
    --require-gpu --require-ci --min-gpus 4
echo "=== Installation complete ==="
python3 -c "import sglang_fl; print(f'sglang_fl {sglang_fl.__name__} loaded')"
