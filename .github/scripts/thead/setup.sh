#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Install sglang-plugin-FL and test dependencies on T-Head PPU.
# SGLang 0.5.18 (srt_empty) + PPU torch 2.10 + FlagTree + sgl-kernel ppu build + FlagGems +
# FlagCX are preinstalled in the CI image (see docker/thead/empty-0.5.18.containerfile);
# this script only reinstalls the plugin from the checkout under test.
# --no-deps: the plugin's dependencies are already in the image, and pip must not
# resolve against (and replace) the vendor torch / FlagTree builds.
set -euo pipefail
git config --global --add safe.directory "$(pwd)"
echo "=== Installing sglang-plugin-FL (T-Head PPU) ==="
pip install --upgrade pip "setuptools>=68,<82" wheel
pip install --no-deps --no-build-isolation -e .
pip install pytest pytest-timeout pyyaml
echo "=== Installation complete ==="
python -c "import importlib.metadata as m, sglang_fl; print('sglang', m.version('sglang'), '| sglang_fl loaded from', sglang_fl.__file__)"
