#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Prepare the NVIDIA two-wheel image for CI validation.

set -euo pipefail
git config --global --add safe.directory "$(pwd)"

echo "=== Activating the Runtime's Triton 3.7 compiler slot ==="
# The operations Runtime keeps compiler implementations in side directories.
# Its official activation command selects /opt/triton (3.7.1), matching the
# pip distribution required by torch/SGLang. Do not switch CI back to the
# FlagTree 3.6 compatibility slot.
source /etc/profile.d/zz-flagos.sh
compiler triton
echo "compiler slot active: PYTHONPATH=$PYTHONPATH"
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "PYTHONPATH=$PYTHONPATH" >> "$GITHUB_ENV"
fi

echo "=== Installing test runner (ephemeral) ==="
python3 -m pip install --break-system-packages \
    pytest==9.1.1 pytest-timeout==2.4.0

# SGLang and sglang-plugin-FL are real wheels baked into the image. The plugin
# activates its cache-local CUDA compatibility view inside every Python process
# before FlashInfer first asks for its JIT toolchain; setup.sh must not patch
# CUDA_HOME, create symlinks, or replace the installed plugin with an editable
# checkout.
python3 .github/scripts/nvidia/verify_environment.py \
    --require-gpu --require-ci --min-gpus 4
echo "=== Installation complete ==="
