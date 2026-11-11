#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Install the PR checkout and ephemeral test tools on T-Head PPU.
# The image contains the operations runtime plus SGLang srt_empty only. The
# runtime constraints generated while building the image forbid CI tooling from
# changing any runtime-owned distribution.
set -euo pipefail
git config --global --add safe.directory "$(pwd)"
echo "=== Installing sglang-plugin-FL (T-Head PPU) ==="
pip install --no-deps --no-build-isolation -e .
pip install -c /opt/runtime-constraints.txt pytest pytest-timeout
echo "=== Installation complete ==="
python -c "import importlib.metadata as m, sglang_fl, yaml; print('sglang', m.version('sglang'), '| sglang_fl loaded from', sglang_fl.__file__)"
# Model store inventory: e2e cases resolve /data/models/Qwen/<model> from the
# host store bind-mounted here; log what is actually visible so a "Model not
# found" collection failure can be told apart from a wrong mount level.
echo "=== Model store inventory (/data/models/Qwen) ==="
ls -la /data/models/Qwen/ 2>&1 || true
