#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Torch / FlagTree / FlagGems / FlagCX ship in the CI image; this script only
# refreshes the editable plugin from the checked-out commit.
set -euo pipefail
git config --global --add safe.directory "$(pwd)"
echo "=== Installing sglang-plugin-FL (Kunlunxin) ==="

PIP_EXTRA=()

python3 -m pip install "${PIP_EXTRA[@]}" --no-deps --no-build-isolation -e .
# CI image should already ship pytest; install lightly if missing.
python3 -m pip install "${PIP_EXTRA[@]}" --no-deps --no-build-isolation \
    pytest pytest-timeout==2.4.0 pyyaml >/dev/null

python3 .github/scripts/kunlunxin/verify_environment.py \
    --check-versions \
    --require-gpu --min-gpus 4
echo "=== Installation complete ==="
python3 -c "import sglang_fl; print(f'sglang_fl {sglang_fl.__name__} loaded')"
