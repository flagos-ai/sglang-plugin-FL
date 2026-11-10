#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# The operations-runtime CI image ships the vendor stack, the SGLang wheel,
# and its build-time evidence pack under /opt/sglang-ci; the plugin and the
# test runner install into the ephemeral job container.
set -euo pipefail
PLUGIN_ROOT="$(pwd)"
# checkout's safe.directory uses a temporary HOME that build subprocesses do
# not inherit. Trust only this checkout for editable metadata's Git inspection.
git config --global --add safe.directory "$PLUGIN_ROOT"
EVIDENCE="${HYGON_RUNTIME_EVIDENCE:-/opt/sglang-ci}"
if [ ! -f "$EVIDENCE/runtime-distributions-before.json" ]; then
  echo "::error::CI image lacks its runtime evidence pack: $EVIDENCE" >&2
  exit 1
fi
# The operations runtime intentionally ships no test packages; install the
# pinned runner into the ephemeral job container, then verify the runner
# landed (check.sh's --check-image runs earlier and cannot see it yet).
python -m pip install pytest==9.1.1 pytest-timeout==2.4.0 \
  iniconfig==2.3.1 pluggy==1.7.0
python .github/scripts/hygon/ci_environment.py --check-ci-packages
# Never resolve dependencies: xgrammar pulls a standalone triton wheel whose
# import is shadowed by FlagTree's /opt/flagtree entry, and the runtime's
# own distributions must never move.
python .github/scripts/hygon/ci_environment.py --check-image -- \
  python -m pip install --no-deps --no-build-isolation -e "$PLUGIN_ROOT"
export PYTHONPATH="$PLUGIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${GITHUB_ENV:-}" ]; then
  printf 'PYTHONPATH=%s\n' "$PYTHONPATH" >> "$GITHUB_ENV"
fi
python .github/scripts/hygon/ci_environment.py -- \
  python .github/scripts/hygon/verify_environment.py \
    --plugin-root "$PLUGIN_ROOT" --runtime-evidence "$EVIDENCE" \
    --require-gpu --min-gpus 2
