#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Install sglang-plugin-FL and test dependencies on Enflame GCU.
# sglang 0.5.11 + torch 2.11 (torch_gcu) + triton 3.6.0 + flag_gems v5.3.0 are
# preinstalled in the CI image (see docker/enflame/containerfile); this script
# only installs the plugin itself.
set -euo pipefail
git config --global --add safe.directory "$(pwd)"
echo "=== Installing sglang-plugin-FL (Enflame GCU) ==="
pip install --upgrade pip "setuptools>=68,<82" wheel
pip install -e ".[dev]" --no-build-isolation || pip install -e . --no-build-isolation
pip install pytest pytest-timeout pyyaml
echo "=== Installation complete ==="
python -c "import sglang_fl; print(f'sglang_fl {sglang_fl.__name__} loaded')"

# Pin the plugin dispatch config to this platform's yaml via SGLANG_FL_CONFIG
# (highest-priority override). Importing sglang.srt flips
# torch.cuda.is_available() to True on GCU, which would otherwise make the
# auto-detection load nvidia.yaml instead of enflame.yaml.
if [[ -n "${GITHUB_ENV:-}" ]]; then
  CFG_PATH="$(python -c \
    'import os, sglang_fl.dispatch.config as c; print(os.path.join(os.path.dirname(c.__file__), "enflame.yaml"))')"
  echo "SGLANG_FL_CONFIG=${CFG_PATH}" >> "${GITHUB_ENV}"
  echo "SGLANG_FL_CONFIG=${CFG_PATH}"
fi
