#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Install sglang-plugin-FL and test dependencies on Enflame GCU.
# The validated runtime image already provides sglang 0.5.11, torch_gcu,
# FlagGems, FlagCX and their hardware runtime dependencies.
set -euo pipefail
git config --global --add safe.directory "$(pwd)"
echo "=== Installing sglang-plugin-FL (Enflame GCU) ==="
pip install "setuptools>=68,<82" wheel
pip install -e ".[dev]" --no-build-isolation || pip install -e . --no-build-isolation
pip install pytest pytest-timeout pyyaml

# The runner injects Enflame's internal package index, which currently does
# not mirror soundfile. New images already contain the SGLang-pinned version;
# refresh older images from the public index configured by the base image.
python -c 'import importlib.metadata; assert importlib.metadata.version("soundfile") == "0.13.1"' || \
    pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple soundfile==0.13.1

echo "=== Installation complete ==="
python -c "import sglang_fl; print(f'sglang_fl {sglang_fl.__name__} loaded')"
