#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Install sglang-plugin-FL and test dependencies on NVIDIA CUDA.
set -euo pipefail
git config --global --add safe.directory "$(pwd)"
echo "=== Installing sglang-plugin-FL (CUDA) ==="
pip install --upgrade pip "setuptools>=68,<82" wheel
pip install -e ".[dev]" --no-build-isolation || pip install -e . --no-build-isolation
pip install pytest pytest-timeout pyyaml
echo "=== Installation complete ==="
python -c "import sglang_fl; print(f'sglang_fl {sglang_fl.__name__} loaded')"
