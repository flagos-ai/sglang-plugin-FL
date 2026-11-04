#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Check MetaX C550 availability, pick four idle cards for the TP4 cases, and
# fail fast on a broken image unpack before any test runs.
set -euo pipefail
echo "=== Checking MetaX GPU availability ==="
mx-smi

CUDA_VISIBLE_DEVICES=$(python .github/scripts/metax/select_gpus.py)
export CUDA_VISIBLE_DEVICES
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
if [ -n "${GITHUB_ENV:-}" ]; then
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES" >> "$GITHUB_ENV"
fi

python - <<'PY'
import torch

# Some hosts unpack this image family without applying its layer whiteouts, which
# leaves stale numpy 2.x / transformers files behind and breaks scipy at import
# time. Surface that here instead of as an unrelated failure deep in a test.
try:
    import numpy
    import scipy.linalg  # noqa: F401
except ImportError as error:
    raise SystemExit(
        f"numpy/scipy import failed ({error}); the CI image was probably unpacked "
        "incompletely on this host (stale files from an overwritten layer). "
        "Re-pull the image on the runner."
    )

assert torch.cuda.is_available(), "MetaX torch cannot access the runner GPUs"
count = torch.cuda.device_count()
assert count >= 4, f"The MetaX TP4 CI cases require four visible GPUs, found {count}"
print(f"torch {torch.__version__}, numpy {numpy.__version__}: {count} visible GPUs")
print(f"GPU 0: {torch.cuda.get_device_name(0)}")
PY
