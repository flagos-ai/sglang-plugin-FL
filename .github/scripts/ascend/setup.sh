#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Install only the checked-out plugin on Huawei Ascend NPU. The versioned CI
# image owns SGLang, CANN, torch_npu, FlagGems, FlagCX, kernels, and test tools.
set -euo pipefail
git config --global --add safe.directory "$(pwd)"
echo "=== Installing sglang-plugin-FL (Ascend) ==="
python3 -m pip install --no-deps --no-build-isolation -e .
# An older plugin is present in the empty base image. Keep this checkout first
# for this step and all later Actions steps without re-resolving dependencies.
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
if [ -n "${GITHUB_ENV:-}" ]; then
  printf 'PYTHONPATH=%s\n' "$PYTHONPATH" >> "$GITHUB_ENV"
fi
python3 .github/scripts/ascend/verify_environment.py \
  --require-ci \
  --require-npu \
  --min-npus 4 \
  --plugin-root "$(pwd)"
echo "=== Installation complete ==="
