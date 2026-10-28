#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Launch MiniMax-M3 BF16 on two Moore Threads S5000 nodes (8 devices per node).
# Run the same script on both nodes with NODE_RANK set to 0 or 1.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the MiniMax-M3 weight directory}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the rank-0 host address}"
: "${NODE_RANK:?Set NODE_RANK to 0 or 1}"
: "${FLAGCX_PATH:?Set FLAGCX_PATH to a built FlagCX checkout}"
: "${EVIDENCE_DIR:?Set EVIDENCE_DIR to a persistent log directory}"

if [[ "${NODE_RANK}" != "0" && "${NODE_RANK}" != "1" ]]; then
  echo "NODE_RANK must be 0 or 1" >&2
  exit 2
fi

mkdir -p "${EVIDENCE_DIR}"
export FLAGCX_PATH
export SGLANG_PLUGINS="${SGLANG_PLUGINS:-sglang_fl}"
export USE_FLAGGEMS="${USE_FLAGGEMS:-1}"
export SGLANG_FL_PER_OP="${SGLANG_FL_PER_OP:-topk=vendor}"
export SGLANG_ROPE_CACHE_FP32="${SGLANG_ROPE_CACHE_FP32:-1}"
export SGLANG_FLAGGEMS_RECORD=1
export SGLANG_FLAGGEMS_LOG_PATH="${EVIDENCE_DIR}/flaggems-rank${NODE_RANK}.log"
export SGLANG_FL_DISPATCH_DEBUG=1
export SGLANG_FL_DISPATCH_LOG="${EVIDENCE_DIR}/dispatch-rank${NODE_RANK}.log"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"
export FLAGCX_SOCKET_IFNAME="${FLAGCX_SOCKET_IFNAME:-${GLOO_SOCKET_IFNAME}}"

exec python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME:-MiniMax-M3}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --tp-size 8 \
  --pp-size 2 \
  --pp-async-batch-depth 1 \
  --pp-max-micro-batch-size 64 \
  --nnodes 2 \
  --node-rank "${NODE_RANK}" \
  --dist-init-addr "${MASTER_ADDR}:${DIST_PORT:-31019}" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-31018}" \
  --context-length "${CONTEXT_LENGTH:-8192}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-8192}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.85}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-64}" \
  --attention-backend triton \
  --moe-runner-backend triton \
  --cuda-graph-backend-decode full \
  --cuda-graph-bs-decode 1 2 4 8 12 16 24 32 40 \
  --cuda-graph-backend-prefill disabled \
  --disable-overlap-schedule
