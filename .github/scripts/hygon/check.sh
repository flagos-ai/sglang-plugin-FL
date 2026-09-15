#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Check Hygon DCU availability.
set -euo pipefail
echo "=== Checking Hygon DCU availability ==="

HY_SMI_BIN=""
if command -v hy-smi >/dev/null 2>&1; then
  HY_SMI_BIN="$(command -v hy-smi)"
elif [[ -x /opt/hyhal/bin/hy-smi ]]; then
  HY_SMI_BIN="/opt/hyhal/bin/hy-smi"
fi

if [[ -n "${HY_SMI_BIN}" ]]; then
  echo "Using hy-smi: ${HY_SMI_BIN}"
  "${HY_SMI_BIN}"
else
  echo "::warning::hy-smi not found in PATH or /opt/hyhal/bin; skipping SMI output."
fi

test -e /dev/kfd
test -e /dev/mkfd
test -d /dev/dri
test -d /opt/hyhal

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("Hygon DCU (torch.cuda via HIP) is not available")
count = torch.cuda.device_count()
print(f"DCU count: {count}")
if count < 4:
    raise RuntimeError(f"At least 4 DCUs are required for tp4 cases, found {count}")
PY
