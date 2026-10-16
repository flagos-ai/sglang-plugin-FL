#!/usr/bin/env bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Validate the reserved CoreX allocation without selecting or evicting GPUs.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
mkdir -p artifacts/iluvatar

check_corex() {
    echo "=== Checking the explicitly allocated Iluvatar CoreX GPUs ==="
    if ! [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "CUDA_VISIBLE_DEVICES must explicitly list reserved numeric GPU indices" >&2
        return 2
    fi
    export SGLANG_ILUVATAR_MIN_DEVICES="${SGLANG_ILUVATAR_MIN_DEVICES:-4}"
    if ! [[ "$SGLANG_ILUVATAR_MIN_DEVICES" =~ ^[1-9][0-9]*$ ]]; then
        echo "SGLANG_ILUVATAR_MIN_DEVICES must be a positive integer" >&2
        return 2
    fi
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; required devices=$SGLANG_ILUVATAR_MIN_DEVICES"

    # Fail closed on unavailable inventory or busy memory before importing torch
    # or initializing a context on any allocated device. Never choose other GPUs.
    python scripts/iluvatar/require_idle_gpus.py
    python - <<'PY'
import os

import torch

selected = [int(index) for index in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
minimum = int(os.environ["SGLANG_ILUVATAR_MIN_DEVICES"])
assert hasattr(torch, "corex"), "A real Iluvatar CoreX torch build is required"
assert torch.cuda.is_available(), "CoreX must expose its available CUDA-alias runtime"
count = torch.cuda.device_count()
assert count == len(selected), (
    f"CUDA_VISIBLE_DEVICES allocated {len(selected)} GPUs, but torch sees {count}; "
    "refusing to test an unexpected device allocation"
)
assert count >= minimum, f"Need {minimum} reserved CoreX GPUs, found {count}"
print(f"Torch {torch.__version__}: {count} explicitly allocated CoreX GPUs")
for logical, physical in enumerate(selected):
    tensor = torch.arange(4, device=f"cuda:{logical}", dtype=torch.float32)
    actual = tensor.square().sum().item()
    assert actual == 14.0, f"Tensor arithmetic failed on logical cuda:{logical}: {actual}"
    torch.cuda.synchronize(logical)
    print(
        f"cuda:{logical} -> allocated physical GPU {physical}: "
        f"{torch.cuda.get_device_name(logical)}; tensor smoke={actual}"
    )
PY
    if [[ -n "${GITHUB_ENV:-}" ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%s\nSGLANG_ILUVATAR_MIN_DEVICES=%s\n' \
            "$CUDA_VISIBLE_DEVICES" "$SGLANG_ILUVATAR_MIN_DEVICES" >> "$GITHUB_ENV"
    fi
}

check_corex 2>&1 | tee artifacts/iluvatar/check.log
