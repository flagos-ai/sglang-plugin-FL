#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# The versioned CI image provides patched SGLang and the captured vendor stack.
set -euo pipefail
PLUGIN_ROOT="$(pwd)"
BASELINE="${HYGON_VENDOR_BASELINE:-/opt/hygon-vendor-baseline.json}"
if [ ! -f "$BASELINE" ]; then
  echo "::error::Hygon CI image lacks its captured vendor baseline: $BASELINE" >&2
  exit 1
fi
# Reject main's old image before installation. Never resolve dependencies:
# xgrammar can otherwise replace FlagTree with standalone Triton.
python .github/scripts/hygon/ci_environment.py --check-image -- \
  python -m pip install --no-deps --no-build-isolation -e "$PLUGIN_ROOT"
export PYTHONPATH="$PLUGIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${GITHUB_ENV:-}" ]; then
  printf 'PYTHONPATH=%s\n' "$PYTHONPATH" >> "$GITHUB_ENV"
fi
python .github/scripts/hygon/ci_environment.py -- \
  python .github/scripts/hygon/verify_environment.py \
    --plugin-root "$PLUGIN_ROOT" --baseline "$BASELINE" --require-gpu --min-gpus 2
