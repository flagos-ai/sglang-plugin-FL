#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Install the checked-out plugin into the TsingMicro SGLang 0.5.18 CI image.
# The image (docker/tsingmicro/empty-0.5.18.containerfile) supplies SGLang and
# the TXDA vendor stack; install only this checkout, never resolve deps.
set -euo pipefail
PLUGIN_ROOT="$(pwd)"
git config --global --add safe.directory "$PLUGIN_ROOT"
echo "=== Installing sglang-plugin-FL (TsingMicro) ==="
python3 -m pip install --no-deps --no-build-isolation -e "$PLUGIN_ROOT"
# Put this checkout ahead of the copy baked into the image, also for later
# Actions steps and their subprocesses.
export PYTHONPATH="$PLUGIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${GITHUB_ENV:-}" ]; then
  printf 'PYTHONPATH=%s\n' "$PYTHONPATH" >> "$GITHUB_ENV"
fi
# Rejects a wrong image (e.g. SGLang 0.5.11, a second environment, CUDA wheels).
python3 docker/tsingmicro/verify_environment.py --plugin-root "$PLUGIN_ROOT" \
  --require-device --min-cards 8
echo "=== Installation complete ==="
