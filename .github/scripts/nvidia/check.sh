#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Check NVIDIA GPU availability.
set -euo pipefail
echo "=== Checking NVIDIA GPU availability ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "nvidia-smi is not installed in the runtime image; using PyTorch CUDA checks"
fi
# The current CUDA E2E matrix contains TP=4 cases, so fail early when a runner
# exposes too few GPUs rather than timing out during model launch.
# --require-ci is intentionally omitted: check.sh runs before setup.sh
# installs the pinned test runner into the ephemeral container (the
# operations-runtime image ships no test packages); setup.sh verifies the
# runner afterwards.
python3 .github/scripts/nvidia/verify_environment.py \
    --require-gpu --min-gpus 4
