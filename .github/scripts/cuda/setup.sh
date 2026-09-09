#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Install sglang-plugin-FL and test dependencies on NVIDIA CUDA.
set -euo pipefail
git config --global --add safe.directory "$(pwd)"
echo "=== Installing sglang-plugin-FL (CUDA) ==="
# The upstream runtime is based on Ubuntu 24.04 and marks its system Python as
# externally managed (PEP 668). This is an ephemeral CI container, so install
# only the editable plugin and explicitly preserve every pinned dependency.
python3 -m pip install \
    --break-system-packages --no-deps --no-build-isolation -e .
python3 .github/scripts/cuda/verify_environment.py \
    --require-gpu --require-ci --min-gpus 4
echo "=== Installation complete ==="
python3 -c "import sglang_fl; print(f'sglang_fl {sglang_fl.__name__} loaded')"
