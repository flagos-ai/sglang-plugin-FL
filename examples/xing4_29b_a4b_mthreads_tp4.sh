#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the Xing4.0-29B-A4B checkpoint directory}"

export SGLANG_PLUGINS=sglang_fl
export SGLANG_FL_MODEL_VENDOR=mthreads
export USE_FLAGGEMS="${USE_FLAGGEMS:-1}"
export SGLANG_FL_DIST_BACKEND="${SGLANG_FL_DIST_BACKEND:-flagcx}"

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME:-Xing4.0-29B-A4B}" \
  --tp-size "${TP_SIZE:-4}" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-31082}" \
  --trust-remote-code \
  --context-length "${CONTEXT_LENGTH:-32768}" \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --reasoning-parser xingchen4 \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-64}"
