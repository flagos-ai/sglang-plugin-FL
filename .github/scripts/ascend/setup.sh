#!/bin/bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Prepare the Ascend two-wheel CI image for testing.
#
# The image ships the two application wheels (sglang, sglang_fl); this job
# installs only the pinned test runner (test-requirements.txt) into the
# ephemeral container and proves the installed-wheel contract
# (verify_environment.py). The plugin checkout under test is never
# installed: tests exercise the installed wheel, and tests/run.py stages
# every pytest child outside the checkout.
#
# The Runtime's Bash environment (BASH_ENV=/etc/bash_env.sh) exposes the
# /opt/flagtree compiler slot; tests/run.py keeps that entry while dropping
# any checkout path.
set -euo pipefail

PLUGIN_ROOT="$(pwd)"
VERIFY=.github/scripts/ascend/verify_environment.py

git config --global --add safe.directory "$PLUGIN_ROOT"

# The Runtime's PYTHONPATH ends with an empty entry (cwd precedence), which
# would let the checkout shadow the installed wheel for every plain python
# invocation from the repo root. Strip empty entries; the compiler slot and
# the CANN paths stay. tests/run.py applies the same filtering to every
# pytest child.
export PYTHONPATH="$(python -c 'import os; print(":".join(e for e in os.environ.get("PYTHONPATH", "").split(":") if e))')"

# Job-time drift reference: the distribution inventory before the install.
SNAPSHOT="$(mktemp /tmp/ascend-runtime-snapshot.XXXXXX)"
trap 'rm -f "$SNAPSHOT"' EXIT
python "$PLUGIN_ROOT/$VERIFY" \
  --capture-runtime-snapshot "$SNAPSHOT" --plugin-root "$PLUGIN_ROOT"

# The operations runtime ships no test packages. uv and the virtualenv live
# at the paths the Runtime itself ships (/usr/local/bin/uv, /flagos).
/usr/local/bin/uv pip install --python /flagos/bin/python3 \
  --no-cache \
  -r "$PLUGIN_ROOT/.github/scripts/ascend/test-requirements.txt"

python "$PLUGIN_ROOT/$VERIFY" \
  --verify --runtime-evidence "$SNAPSHOT" --plugin-root "$PLUGIN_ROOT"

# Every later pytest child tests the installed wheel; tests/run.py keeps the
# default checkout mode when this is unset.
export SGLANG_FL_TEST_SOURCE=wheel
if [ -n "${GITHUB_ENV:-}" ]; then
  printf 'SGLANG_FL_TEST_SOURCE=wheel\n' >> "$GITHUB_ENV"
fi
echo "=== Ascend setup complete ==="
