#!/bin/bash
# Copyright 2026 BAAI. All rights reserved.
# Prepare the Hygon two-wheel image for CI validation.
#
# The image ships the real application wheels (sglang 0.5.18+flagos and
# sglang_fl 0.1.0); this job only installs the pinned test runner and
# verifies the installed-wheel contract. All Python verification goes
# through .github/scripts/hygon/verify_environment.py (snapshot capture
# plus the post-setup checks); the checkout source tree must never shadow
# the installed plugin - SGLANG_FL_TEST_SOURCE=wheel makes tests/run.py
# stage every pytest child (and its engine grandchildren) from an
# isolated directory with PYTHONSAFEPATH on.
set -euo pipefail
PLUGIN_ROOT="$(pwd)"
# checkout's safe.directory uses a temporary HOME that build subprocesses do
# not inherit. Trust only this checkout for editable metadata's Git inspection.
git config --global --add safe.directory "$PLUGIN_ROOT"

# Job-time drift reference: record the pristine distribution inventory
# BEFORE installing the pinned test runner. Image-level zero-drift is
# proven at build time by docker/hygon/scripts/verify_image.sh; CI
# consumes the accepted image by digest, so this snapshot only anchors
# drift introduced after container start.
RUNTIME_SNAPSHOT="$(mktemp /tmp/hygon-runtime-snapshot.XXXXXX.json)"
python "$PLUGIN_ROOT/.github/scripts/hygon/verify_environment.py" \
  --capture-runtime-snapshot "$RUNTIME_SNAPSHOT"

# The operations runtime intentionally ships no test packages; install the
# pinned runner (single source: test-requirements.txt) into the ephemeral
# job container, then verify it landed (check.sh's --check-image runs
# earlier and cannot see it yet). The runner is generic packages only, so
# the Tencent mirror is the sole index: --default-index replaces uv's
# default index outright (a plain --index extra would keep the default as
# the fallback), matching the image build's network policy.
/usr/local/bin/uv pip install --python /flagos/bin/python3 \
  --default-index https://mirrors.cloud.tencent.com/pypi/simple \
  -r "$PLUGIN_ROOT/.github/scripts/hygon/test-requirements.txt"
python .github/scripts/hygon/ci_environment.py --check-ci-packages

# Select the installed-wheel test mode for every later pytest subprocess;
# tests/run.py keeps the default checkout mode when this is unset.
export SGLANG_FL_TEST_SOURCE=wheel
if [ -n "${GITHUB_ENV:-}" ]; then
  printf 'SGLANG_FL_TEST_SOURCE=wheel\n' >> "$GITHUB_ENV"
fi

# Diagnostic-only bypass for build-infra#1787. Keep the checked-in model
# contract on strict HCU attention; only e2e construction sees this switch.
if [ -n "${GITHUB_ENV:-}" ]; then
  printf 'SGLANG_HYGON_DIAGNOSTIC_TRITON_ATTENTION=1\n' >> "$GITHUB_ENV"
fi

# The image bakes the DTK prefix contract (ROCM_PATH, the /opt/rocm alias
# and the FlagTree clang wrapper, see the containerfile); this job only
# verifies it. The FlagTree wrapper selection check runs inside
# verify_environment.py below.
if [ "$(readlink -f /opt/rocm)" != "$(readlink -f /opt/dtk)" ]; then
  echo "::error::/opt/rocm is missing or does not resolve to the DTK SDK" >&2
  exit 1
fi
/opt/rocm/bin/hipcc --version | head -n 1

python .github/scripts/hygon/ci_environment.py -- \
  python .github/scripts/hygon/verify_environment.py \
    --plugin-root "$PLUGIN_ROOT" --plugin-source wheel \
    --runtime-evidence "$RUNTIME_SNAPSHOT" \
    --require-gpu --min-gpus 2
