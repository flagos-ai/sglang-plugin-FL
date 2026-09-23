#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Check Huawei Ascend NPU availability.
#
# npu-smi is a host-side *driver* tool (/usr/local/Ascend/driver/tools/npu-smi),
# NOT part of the CANN toolkit. The CI image bundles the CANN toolkit; driver
# userspace is bind-mounted from the host by ascend.yml, and the image prepends
# /usr/local/Ascend/driver/tools to PATH, so npu-smi is
# normally on PATH. This script still handles the off-PATH/missing case (driver
# mount regressed, or an older image without the PATH entry); the pinned
# torch_npu verifier below remains the authoritative availability signal.
#
# But a missing npu-smi CLI does NOT mean the NPU is unusable: torch_npu reaches
# the device via /dev/davinci* nodes + driver libs, independent of the CLI. So
# this script:
#   1. Diagnoses the environment (paths, device nodes, npu-smi location).
#   2. Recovers by extending PATH if npu-smi exists off-PATH.
#   3. Runs the pinned environment verifier and requires four working NPUs.
#
# NOTE: `set -e` is intentionally omitted so all diagnostics run before we decide.
set -uo pipefail

echo "=== Checking Ascend NPU availability ==="

# ---------------------------------------------------------------------------
# 1. Environment & expected Ascend paths
# ---------------------------------------------------------------------------
echo "--- Environment ---"
echo "whoami=$(whoami)  hostname=$(hostname)"
echo "PATH=$PATH"
echo "ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-<unset>}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

echo "--- Ascend directories ---"
for d in \
  /usr/local/Ascend \
  /usr/local/Ascend/ascend-toolkit/latest \
  /usr/local/Ascend/driver \
  /usr/local/Ascend/driver/tools \
  /usr/local/Ascend/driver/lib64 \
  /usr/local/sbin ; do
  if [ -e "$d" ]; then
    echo "[exists] $d"
  else
    echo "[absent] $d"
  fi
done

# ---------------------------------------------------------------------------
# 2. Device nodes (injected by the Ascend OCI runtime configured in ascend.yml)
# ---------------------------------------------------------------------------
echo "--- Device nodes ---"
for dev in \
  /dev/davinci0 /dev/davinci1 /dev/davinci2 /dev/davinci3 \
  /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
  if [ -e "$dev" ]; then
    echo "[exists] $dev"
  else
    echo "[absent]  $dev"
  fi
done

# ---------------------------------------------------------------------------
# 3. Locate npu-smi (PATH first, then known driver locations)
# ---------------------------------------------------------------------------
echo "--- Locate npu-smi ---"
NPU_SMI=""
path_hit="$(command -v npu-smi 2>/dev/null || true)"
if [ -n "$path_hit" ] && [ -x "$path_hit" ] && [ -s "$path_hit" ]; then
  NPU_SMI="$path_hit"
  echo "found on PATH: $NPU_SMI"
else
  if [ -n "$path_hit" ]; then
    echo "ignoring unusable PATH entry: $path_hit (not executable or empty)"
  fi
  echo "not on PATH; searching known driver locations..."
  for cand in \
    /usr/local/Ascend/driver/tools/npu-smi \
    /usr/local/Ascend/driver/usr/local/sbin/npu-smi \
    /usr/local/bin/npu-smi \
    /usr/local/sbin/npu-smi \
    /usr/bin/npu-smi ; do
    if [ -x "$cand" ] && [ -s "$cand" ]; then
      NPU_SMI="$cand"
      echo "found at: $NPU_SMI  (off-PATH)"
      break
    fi
    echo "not at: $cand"
  done
fi

# ---------------------------------------------------------------------------
# 4. Run npu-smi (extend PATH if found off-PATH)
# ---------------------------------------------------------------------------
npu_smi_ok=0
if [ -n "$NPU_SMI" ]; then
  smi_dir="$(dirname "$NPU_SMI")"
  case ":$PATH:" in
    *":$smi_dir:"*) ;;
    *) export PATH="$smi_dir:$PATH"; echo "extended PATH with $smi_dir" ;;
  esac
  echo "--- npu-smi info ---"
  if "$NPU_SMI" info; then
    npu_smi_ok=1
  else
    rc=$?
    echo "WARN: npu-smi found at $NPU_SMI but 'npu-smi info' failed (rc=$rc)"
  fi
else
  echo "WARN: npu-smi binary not found in container."
  echo "      Driver tools not mounted / not on PATH. This alone does NOT block CI"
  echo "      if torch_npu can still reach the NPU (see probe below)."
fi

# ---------------------------------------------------------------------------
# 5. Verify the complete pinned runtime and require the TP=4 device allocation.
# ---------------------------------------------------------------------------
echo "--- Pinned runtime and torch_npu probe ---"
verify_rc=0
python3 .github/scripts/ascend/verify_environment.py \
  --require-ci \
  --require-npu \
  --min-npus 4 || verify_rc=$?

if [ "$verify_rc" -ne 0 ]; then
  echo "FAIL: pinned Ascend environment verification failed (rc=$verify_rc)."
  exit "$verify_rc"
fi

echo "PASS: pinned Ascend runtime and four NPUs are available (npu-smi=$npu_smi_ok)"
