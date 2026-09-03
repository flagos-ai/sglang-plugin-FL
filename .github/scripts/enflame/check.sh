#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Check Enflame GCU availability.
set -euo pipefail
echo "=== Checking Enflame GCU availability ==="
if ! command -v efsmi >/dev/null 2>&1; then
  echo "::error::efsmi is not available in the CI container."
  exit 1
fi
efsmi

python - <<'PY'
import torch
import torch_gcu  # noqa: F401

if not hasattr(torch, "gcu") or not torch.gcu.is_available():
    raise RuntimeError("Enflame GCU is not available")

count = torch.gcu.device_count()
print(f"Enflame GCU count: {count}")
if count < 4:
    raise RuntimeError(f"At least 4 GCUs are required for tp4 cases, found {count}")
PY
