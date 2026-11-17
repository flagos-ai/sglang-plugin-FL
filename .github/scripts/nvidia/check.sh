#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Pre-setup gate: runs before setup.sh and must pass on the untouched
# two-wheel image. Checks the runner's GPUs and the Runtime/wheel identity
# only -- the Triton slot is not activated yet, the test runner is not
# installed yet, and nothing here may mutate the image or the process.
set -euo pipefail
echo "=== Pre-setup check: NVIDIA GPU and image identity ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "nvidia-smi is not installed in the runtime image; using PyTorch CUDA checks"
fi
# The current CUDA E2E matrix contains TP=4 cases, so fail early when a runner
# exposes too few GPUs rather than timing out during model launch.
python3 .github/scripts/nvidia/verify_environment.py \
    --phase pre-setup --min-gpus 4
