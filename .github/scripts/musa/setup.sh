#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Install sglang-plugin-FL and test dependencies on Moore Threads MUSA.
# The versioned MUSA CI image supplies SGLang, vendor torch, FlagTree/FlagGems
# and test dependencies. Install only the checked-out plugin, preserving that stack.
set -euo pipefail
git config --global --add safe.directory "$(pwd)"
echo "=== Installing sglang-plugin-FL (MUSA) ==="
python -m pip install --no-deps --no-build-isolation -e .
# The vendor image also contains an older system-level plugin. Put this checkout
# ahead of it, including in subsequent Actions steps and their subprocesses.
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
if [ -n "${GITHUB_ENV:-}" ]; then
  printf 'PYTHONPATH=%s\n' "$PYTHONPATH" >> "$GITHUB_ENV"
fi
echo "=== Installation complete ==="
python - <<'PY'
from pathlib import Path
import importlib.metadata as metadata
import flag_gems
import sglang_fl
import torch_musa
import triton

assert Path(sglang_fl.__file__).resolve().parent == Path.cwd() / "sglang_fl"
for package in ("sglang", "torch", "torch_musa", "flagtree", "flag_gems"):
    print(f"{package}: {metadata.version(package)}")
print(f"triton: {triton.__version__}")
print(f"FlagGems source: {flag_gems.__file__}")
print(f"Plugin source: {sglang_fl.__file__}")
PY
