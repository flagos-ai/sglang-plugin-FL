#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Check Huawei Ascend NPU availability.
#
# npu-smi is a host-side *driver* tool (/usr/local/Ascend/driver/tools/npu-smi),
# NOT part of the CANN toolkit. The CI image bundles only the toolkit
# (/usr/local/Ascend/ascend-toolkit), so npu-smi is reachable only when the host
# driver dir is volume-mounted into the container AND driver/tools is on PATH.
# ascend.yml currently mounts nothing (container_volumes: []) and the image never
# puts driver/tools on PATH, so "npu-smi: command not found" is expected.
#
# But a missing npu-smi CLI does NOT mean the NPU is unusable: torch_npu reaches
# the device via /dev/davinci* nodes + driver libs, independent of the CLI. So
# this script:
#   1. Diagnoses the environment (paths, device nodes, npu-smi location).
#   2. Recovers by extending PATH if npu-smi exists off-PATH.
#   3. Falls back to a torch_npu probe as the real availability signal.
# It only fails when there is NO evidence the NPU is usable.
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
# 2. Device nodes (passed via --device in ascend.yml)
# ---------------------------------------------------------------------------
echo "--- Device nodes ---"
for dev in /dev/davinci0 /dev/davinci1 /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
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
if [ -n "$path_hit" ]; then
  NPU_SMI="$path_hit"
  echo "found on PATH: $NPU_SMI"
else
  echo "not on PATH; searching known driver locations..."
  for cand in \
    /usr/local/Ascend/driver/tools/npu-smi \
    /usr/local/Ascend/driver/usr/local/sbin/npu-smi \
    /usr/local/sbin/npu-smi \
    /usr/bin/npu-smi ; do
    if [ -x "$cand" ]; then
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
# 5. Fallback: probe the NPU through torch_npu (real availability signal).
#    Matches the project idiom in sglang_fl/dispatch/backends/vendor/ascend/.
# ---------------------------------------------------------------------------
echo "--- torch_npu probe ---"
torch_probe_ok=0
if command -v python3 >/dev/null 2>&1 && python3 - <<'PY' 2>&1; then
import sys
try:
    import torch
    import torch_npu  # noqa: F401  (registers torch.npu)
    avail = torch.npu.is_available()
    count = torch.npu.device_count()
    print(f"torch_npu: is_available={avail} device_count={count}")
    if avail and count > 0:
        try:
            print(f"device0: {torch.npu.get_device_name(0)}")
        except Exception as e:
            print(f"(get_device_name failed: {type(e).__name__}: {e})")
        sys.exit(0)
    sys.exit(1)
except Exception as e:
    print(f"torch_npu probe failed: {type(e).__name__}: {e}")
    sys.exit(1)
PY
  torch_probe_ok=1
else
  echo "WARN: torch_npu probe did not succeed (python3 missing or probe failed)."
fi

# ---------------------------------------------------------------------------
# 6. Verdict
# ---------------------------------------------------------------------------
echo "--- Verdict ---"
if [ "$npu_smi_ok" = "1" ] || [ "$torch_probe_ok" = "1" ]; then
  echo "PASS: NPU available (npu-smi=$npu_smi_ok torch_npu=$torch_probe_ok)"
  exit 0
fi

echo "FAIL: NPU not available - npu-smi missing AND torch_npu probe failed."
echo "      Likely causes (in order of probability):"
echo "        1. Host Ascend driver not mounted into the container."
echo "           Fix: add '/usr/local/Ascend/driver:/usr/local/Ascend/driver'"
echo "                to container_volumes in .github/configs/ascend.yml."
echo "        2. /dev/davinci* device nodes not passed (check --device in ascend.yml)."
echo "        3. Runner host has no Ascend hardware or driver not loaded."
echo "        4. Driver/toolkit version mismatch (image is CANN 8.5.1)."
exit 1
