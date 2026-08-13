#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Install sglang-plugin-FL and test dependencies on Huawei Ascend NPU.
# CANN toolkit + torch_npu are preinstalled in the CI image (see
# docker/ascend/containerfile); this script only installs the plugin itself.
set -euo pipefail
git config --global --add safe.directory "$(pwd)"
echo "=== Installing sglang-plugin-FL (Ascend) ==="
pip install --upgrade pip "setuptools>=68,<82" wheel
pip install -e ".[dev]" --no-build-isolation || pip install -e . --no-build-isolation
pip install pytest pytest-timeout pyyaml
echo "=== Installation complete ==="
python -c "import torch_npu; print(f'torch_npu {torch_npu.__version__} loaded')"
python -c "import sglang_fl; print('sglang_fl plugin loaded')"
