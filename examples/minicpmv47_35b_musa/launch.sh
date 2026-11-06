#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the external MiniCPM-V-4.7-35B-A3B weight directory}"

export SGLANG_PLUGINS=sglang_fl
export USE_FLAGGEMS="${USE_FLAGGEMS:-1}"
export SGLANG_MUSA_FLAGOS_CHUNK_GDN="${SGLANG_MUSA_FLAGOS_CHUNK_GDN:-0}"

exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name MiniCPM-V-4.7-35B-A3B \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-30000}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --tp-size 2 \
  --context-length 262144 \
  --page-size 1 \
  --max-running-requests 16
