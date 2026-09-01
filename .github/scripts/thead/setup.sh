#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Install sglang-plugin-FL and test dependencies on T-Head PPU.
# sglang 0.5.12 + torch 2.10 + sgl-kernel + flag_gems 5.3.0rc2 + flagcx 0.13.0 are
# preinstalled in the CI image (see docker/thead/containerfile); this script only
# installs the plugin itself (the base image's stale sglang_fl is uninstalled at
# build time so this editable install is authoritative).
set -euo pipefail
git config --global --add safe.directory "$(pwd)"
echo "=== Installing sglang-plugin-FL (T-Head PPU) ==="
pip install --upgrade pip "setuptools>=68,<82" wheel
pip install -e ".[dev]" --no-build-isolation || pip install -e . --no-build-isolation
pip install pytest pytest-timeout pyyaml
echo "=== Installation complete ==="
python -c "import sglang_fl; print(f'sglang_fl {sglang_fl.__name__} loaded')"
