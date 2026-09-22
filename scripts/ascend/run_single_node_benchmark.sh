#!/usr/bin/env bash
# Single-node Ascend TP=4 fixed-matrix serving benchmark.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=acceptance_common.sh
source "${SCRIPT_DIR}/acceptance_common.sh"

usage() {
  cat <<'EOF'
Usage: bash scripts/ascend/run_single_node_benchmark.sh [options]

Options:
  --model {all|27b|35b}  Limit the model set (default: all).
  --port PORT             First API port (default: 30000; second model: +10).
  --result-dir PATH       Artifact directory (default: timestamp under /tmp).
  -h, --help              Show this help.

Environment:
  MODEL_27B_PATH          Default: /models/Qwen3.6-27B
  MODEL_35B_PATH          Default: /models/Qwen3.6-35B-A3B
  PYTHON_BIN              Default: python3
  SERVER_START_TIMEOUT    Health-check timeout in seconds (default: 1800)
  MEM_FRACTION_STATIC     Server static-memory fraction (default: 0.80)
  CUDA_GRAPH_MAX_BS_DECODE
                           Maximum decode graph batch size (default: 70)

For every selected model this starts a TP=4, PP=1, single-node server and
runs the fixed acceptance matrix through benchmark_throughput_serve.py:
  (1K, 1K), (4K, 1K), and (16K, 1K), each at 64 requests/concurrency,
  four times with run 1 retained as warmup and runs 2-4 averaged.
Official SGLang JSONL is preserved and validated exactly; a partial request,
token-count mismatch, subprocess error, or missing metric returns non-zero.
EOF
}

MODEL_SELECTION=all
BASE_API_PORT=30000
REQUESTED_RESULT_DIR=""
SERVER_PID=""
SERVER_PGID=""

while (($#)); do
  case "$1" in
    --model)
      (($# >= 2)) || die "--model requires a value"
      MODEL_SELECTION="$2"
      shift 2
      ;;
    --port)
      (($# >= 2)) || die "--port requires a value"
      BASE_API_PORT="$2"
      shift 2
      ;;
    --result-dir)
      (($# >= 2)) || die "--result-dir requires a value"
      REQUESTED_RESULT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "${MODEL_SELECTION}" in
  all|27b|35b) ;;
  *) die "--model must be one of: all, 27b, 35b" ;;
esac
[[ "${BASE_API_PORT}" =~ ^[0-9]+$ ]] \
  && ((BASE_API_PORT > 0 && BASE_API_PORT < 65526)) \
  || die "--port must be an integer in 1..65525"

MODEL_27B_PATH="${MODEL_27B_PATH:-/models/Qwen3.6-27B}"
MODEL_35B_PATH="${MODEL_35B_PATH:-/models/Qwen3.6-35B-A3B}"

require_command "${PYTHON_BIN}"
require_command setsid
require_command ps
require_command awk
if [[ "${MODEL_SELECTION}" == all || "${MODEL_SELECTION}" == 27b ]]; then
  require_directory "${MODEL_27B_PATH}"
fi
if [[ "${MODEL_SELECTION}" == all || "${MODEL_SELECTION}" == 35b ]]; then
  require_directory "${MODEL_35B_PATH}"
fi

configure_performance_mode
require_npus 4
make_result_dir "${REQUESTED_RESULT_DIR}" single_node_benchmark
write_environment_manifest

server_group_has_live_processes() {
  [[ -n "${SERVER_PGID}" ]] || return 1
  ps -eo pgid=,stat= 2>/dev/null \
    | awk -v pgid="${SERVER_PGID}" \
      '$1 == pgid && substr($2, 1, 1) != "Z" { found = 1 } END { exit !found }'
}

stop_server() {
  local attempt
  [[ -n "${SERVER_PID}" ]] || return 0

  if server_group_has_live_processes; then
    log "stopping server process group ${SERVER_PGID}"
    kill -TERM -- "-${SERVER_PGID}" 2>/dev/null || true
    for ((attempt = 0; attempt < 30; attempt++)); do
      server_group_has_live_processes || break
      sleep 1
    done
    if server_group_has_live_processes; then
      log "server group did not stop after 30 seconds; sending SIGKILL"
      kill -KILL -- "-${SERVER_PGID}" 2>/dev/null || true
    fi
  fi

  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
  SERVER_PGID=""
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_benchmark_model() {
  local label="$1"
  local model_path="$2"
  local offset="$3"
  local api_port=$((BASE_API_PORT + offset))
  local server_log="${RESULT_DIR}/${label}_server.log"
  local -a server_cmd=(
    "${PYTHON_BIN}" -m sglang.launch_server
    --model-path "${model_path}"
    --tp-size 4
    --pp-size 1
    --nnodes 1
    --host 0.0.0.0
    --port "${api_port}"
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.80}"
    --max-running-requests 64
    --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE:-70}"
    --attention-backend ascend
    --device npu
    --dtype bfloat16
    --disable-radix-cache
    --trust-remote-code
    --served-model-name "${label}"
  )

  printf '[ascend-acceptance] command:' | tee -a "${server_log}"
  printf ' %q' "${server_cmd[@]}" | tee -a "${server_log}"
  printf '\n' | tee -a "${server_log}"
  setsid "${server_cmd[@]}" >>"${server_log}" 2>&1 &
  SERVER_PID=$!
  SERVER_PGID="${SERVER_PID}"

  log "${label}: server PID ${SERVER_PID}; waiting for health"
  wait_for_health \
    127.0.0.1 "${api_port}" "${SERVER_START_TIMEOUT:-1800}" "${SERVER_PID}"
  kill -0 "${SERVER_PID}" 2>/dev/null \
    || die "${label}: server exited after reporting health; see ${server_log}"

  run_logged "${RESULT_DIR}/${label}_benchmark_driver.log" \
    "${PYTHON_BIN}" "${REPO_ROOT}/benchmarks/benchmark_throughput_serve.py" \
    --model "${model_path}" \
    --model-name "${label}" \
    --host 127.0.0.1 \
    --port "${api_port}" \
    --output-dir "${RESULT_DIR}/${label}_results"

  stop_server
}

if [[ "${MODEL_SELECTION}" == all || "${MODEL_SELECTION}" == 27b ]]; then
  run_benchmark_model qwen3_6_27b "${MODEL_27B_PATH}" 0
fi
if [[ "${MODEL_SELECTION}" == all || "${MODEL_SELECTION}" == 35b ]]; then
  run_benchmark_model qwen3_6_35b_a3b "${MODEL_35B_PATH}" 10
fi

log "PASS: single-node TP=4 benchmark matrix completed; no run was skipped"
