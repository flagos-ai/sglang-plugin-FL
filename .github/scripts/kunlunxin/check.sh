#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Check Kunlunxin XPU availability (CUDA compatibility layer).
set -euo pipefail
echo "=== Checking Kunlunxin XPU availability ==="
# The p800 e2e matrix uses TP=4, so fail early when the runner exposes too few
# devices rather than timing out during model launch.
python3 .github/scripts/kunlunxin/verify_environment.py \
    --check-versions \
    --require-gpu --min-gpus 4
