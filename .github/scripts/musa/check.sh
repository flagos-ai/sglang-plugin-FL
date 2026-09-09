#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Check Moore Threads MUSA GPU availability.
set -euo pipefail
echo "=== Checking Moore Threads MUSA GPU availability ==="
mthreads-gmi
python - <<'PY'
import torch
import torch_musa

assert torch.musa.is_available(), "torch_musa cannot access the runner GPUs"
count = torch.musa.device_count()
assert count >= 4, f"The MUSA TP4 CI cases require four visible GPUs, found {count}"
print(f"torch_musa {torch_musa.__version__}: {count} visible GPUs")
print(f"GPU 0: {torch.musa.get_device_name(0)}")
PY
