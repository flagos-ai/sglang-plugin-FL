#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Install sglang-plugin-FL and test dependencies on Moore Threads MUSA.
# sglang (v0.5.12, srt_musa) + torch_musa + sgl-kernel musa + FlagGems v5.3.1 are
# preinstalled in the CI image (see docker/mthreads/containerfile); this script only
# installs the plugin itself.
set -euo pipefail
git config --global --add safe.directory "$(pwd)"
echo "=== Installing sglang-plugin-FL (MUSA) ==="
pip install --upgrade pip "setuptools>=68,<82" wheel
pip install -e ".[dev]" --no-build-isolation || pip install -e . --no-build-isolation
pip install pytest pytest-timeout pyyaml
echo "=== Installation complete ==="
python -c "import torch_musa; print(f'torch_musa {torch_musa.__version__} loaded')"
python -c "import sglang_fl; print(f'sglang_fl {sglang_fl.__name__} loaded')"
