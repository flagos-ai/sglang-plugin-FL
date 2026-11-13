#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Install only the checked-out plugin on Huawei Ascend NPU. The versioned CI
# image owns SGLang, CANN, torch_npu, FlagGems, FlagCX, kernels, and test tools.
set -euo pipefail

if (( $# > 1 )); then
  echo "Usage: $0 [FlagTree wheel]" >&2
  exit 2
fi
compiler_args=()
if (( $# == 1 )); then
  # This ARM64 wheel targets the CANN 8.5 runtime in the pinned CI image.
  wheel_sha256=1d0c50f4828accc890a83f150d48fc12893a4ef775e8ea7647d73bca8aae5c53
  printf '%s  %s\n' "$wheel_sha256" "$1" | sha256sum --check
  python3 -m pip uninstall -y triton triton-ascend flagtree
  python3 -m pip install --no-deps --no-index "$1"
  compiler_args=(--flagtree)
fi

git config --global --add safe.directory "$(pwd)"
echo "=== Installing sglang-plugin-FL (Ascend) ==="
python3 -m pip install --no-deps --no-build-isolation -e .
# An older plugin is present in the empty base image. Keep this checkout first
# for this step and all later Actions steps without re-resolving dependencies.
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
if [ -n "${GITHUB_ENV:-}" ]; then
  printf 'PYTHONPATH=%s\n' "$PYTHONPATH" >> "$GITHUB_ENV"
fi
python3 .github/scripts/ascend/verify_environment.py \
  --require-ci \
  --require-npu \
  --min-npus 4 \
  --plugin-root "$(pwd)" \
  "${compiler_args[@]}"
echo "=== Installation complete ==="
