#!/bin/bash
# Copyright 2026 BAAI. All rights reserved.
# Prepare the Hygon two-wheel image for CI validation.
#
# The image ships the real application wheels (the derived HCU sglang wheel and
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

# The Runtime owns the DTK environment. BASH_ENV loads vendor.sh before this
# job, so verify the exported root and compiler rather than recreating an SDK
# alias in the application image.
if [ "${ROCM_PATH:-}" != "/opt/dtk" ]; then
  echo "::error::Runtime did not export ROCM_PATH=/opt/dtk" >&2
  exit 1
fi
if [ ! -x "${ROCM_PATH}/bin/hipcc" ]; then
  echo "::error::Runtime did not expose ${ROCM_PATH}/bin/hipcc" >&2
  exit 1
fi
"${ROCM_PATH}/bin/hipcc" --version | head -n 1

python .github/scripts/hygon/ci_environment.py -- \
  python .github/scripts/hygon/verify_environment.py \
    --plugin-root "$PLUGIN_ROOT" --plugin-source wheel \
    --runtime-evidence "$RUNTIME_SNAPSHOT" \
    --require-gpu --min-gpus 2

# TEMP probe for build-infra#1960: identify the host hyhal/driver pair that the
# runner mounts into this job. Strictly read-only; remove this block once the
# Hygon runner driver inventory is recorded.
{
  echo "uname: $(uname -sr)"
  echo "--- hyhal mounts ---"
  grep -E 'hyhal' /proc/mounts 2>/dev/null | head -5 || true
  echo "--- /opt/hyhal ---"
  ls -la /opt/hyhal 2>/dev/null | head -25 || true
  echo "--- /usr/local/hyhal ---"
  ls -la /usr/local/hyhal 2>/dev/null | head -25 || true
  echo "--- hyhal version/release files ---"
  for f in $(
    find /opt/hyhal /usr/local/hyhal -maxdepth 3 -type f \
      \( -iname '*version*' -o -iname 'release' -o -iname '*.release' \) 2>/dev/null \
      | sort | head -10
  ); do
    echo "--- $f"
    head -10 "$f" 2>/dev/null || true
  done || true
  echo "--- hyhal lib sonames ---"
  ls -l /opt/hyhal/lib /opt/hyhal/lib64 /usr/local/hyhal/lib /usr/local/hyhal/lib64 2>/dev/null \
    | grep -E '\.so' | head -30 || true
  lib=""
  for c in /opt/hyhal/lib/libamd_smi.so /opt/hyhal/lib64/libamd_smi.so \
    /usr/local/hyhal/lib/libamd_smi.so /usr/local/hyhal/lib64/libamd_smi.so; do
    if [ -e "$c" ]; then
      lib="$c"
      break
    fi
  done
  if [ -n "$lib" ]; then
    echo "--- $lib ---"
    echo "amdsmi_init occurrences: $(grep -ac amdsmi_init "$lib" 2>/dev/null || echo 0)"
    echo "amdsmi_get_gpu_enumeration_info occurrences: $(grep -ac amdsmi_get_gpu_enumeration_info "$lib" 2>/dev/null || echo 0)"
  else
    echo "no libamd_smi.so under the hyhal mounts"
  fi
  echo "--- kernel modules ---"
  lsmod 2>/dev/null | grep -iE 'hcu|kfd|hygon|rock|dtk|hsdk' || true
  grep -rH . /sys/module/hcu*/version /sys/module/*kfd*/version /sys/module/rock*/version /sys/module/hy*/version 2>/dev/null \
    | head -10 || true
  echo "--- hy-smi ---"
  smi=""
  for c in /usr/local/hyhal/bin/hy-smi /opt/hyhal/bin/hy-smi; do
    if [ -x "$c" ]; then
      smi="$c"
      break
    fi
  done
  if [ -n "$smi" ]; then
    echo "--- $smi -v (timeout 30s) ---"
    timeout 30 "$smi" -v 2>&1 | head -6 || true
  else
    echo "no hy-smi binary under the hyhal mounts; PATH hy-smi: $(command -v hy-smi || echo none)"
  fi
  true
} 2>&1 | sed 's/^/[hyhal-probe] /' || true

# TEMP experiment for the 35B-A3B MoE failure: can FlagGems'
# moe_align_block_size replace the sgl_kernel one on HCU? Read-only probe.
python "$PLUGIN_ROOT/.github/scripts/hygon/probe_moe_align.py" 2>&1 \
  | sed 's/^/[moe-probe-setup] /' || true

# TEMP probe for the token-id-0 output: is the vendor flash_attn paged decode
# kernel numerically right per head_dim / page_size / length / batch?
python "$PLUGIN_ROOT/.github/scripts/hygon/probe_fa256.py" 2>&1 \
  | sed 's/^/[fa-probe-setup] /' || true
