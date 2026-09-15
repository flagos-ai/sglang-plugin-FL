#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Install sglang-plugin-FL and test dependencies on Hygon DCU.
# sglang 0.5.11 + torch 2.10 (DTK 26.04) + triton 3.6.0 + sgl_kernel 0.4.2.post2
# + flag_gems v5.3.0 are preinstalled in the CI image (see
# docker/hygon/containerfile); this script only installs the plugin itself.
set -euo pipefail
git config --global --add safe.directory "$(pwd)"
echo "=== Installing sglang-plugin-FL (Hygon DCU) ==="
pip install --upgrade pip "setuptools>=68,<82" wheel
pip install -e ".[dev]" --no-build-isolation || pip install -e . --no-build-isolation
pip install pytest pytest-timeout pyyaml
echo "=== Installation complete ==="
python -c "import sglang_fl; print(f'sglang_fl {sglang_fl.__name__} loaded')"

# NOTE: no env exports here — the platform env (full op blacklist) lives in
# tests/platforms/hygon.yaml env_defaults, applied per-scope by tests/run.py.
# SGLANG_FL_CONFIG is deliberately not preset anywhere in CI (see the
# platforms yaml header).
