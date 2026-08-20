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

# NOTE: no SGLANG_FL_CONFIG export (an earlier revision did): GITHUB_ENV
# persists into the unit step, and a preset SGLANG_FL_CONFIG changes the
# dispatch manager's resolution enough to break the env-policy unit tests
# (no other platform's setup.sh exports it either). The load-bearing GCU
# fix — the op blacklist — is carried by tests/platforms/enflame.yaml
# env_defaults instead (env outranks any yaml).
