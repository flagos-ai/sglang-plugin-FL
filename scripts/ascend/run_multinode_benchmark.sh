#!/usr/bin/env bash
# Two-node TP=4, PP=2 serving benchmark. Run once on each node.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=acceptance_common.sh
source "${SCRIPT_DIR}/acceptance_common.sh"

usage() {
  cat <<'EOF'
Usage: bash scripts/ascend/run_multinode_benchmark.sh OPTIONS

Required:
  --role {master|worker}  This node's role/rank (master=0, worker=1).
  --master-addr ADDRESS   Routable address of the master node.

Options:
  --model {all|27b|35b}  Limit the model set (default: all).
  --interface NAME        HCCL/Gloo interface (default: business).
  --port PORT             First API port (default: 30000).
  --dist-port PORT        First rendezvous port (default: 21000).
  --nccl-port PORT        First collective port (default: 29765).
  --result-dir PATH       Local artifact directory (default: /tmp timestamp).
  -h, --help              Show this help.

The master runs the fixed acceptance matrix for every selected model:
  (1K, 1K), (4K, 1K), and (16K, 1K), each at 64 requests/concurrency,
  four times with run 1 retained as warmup and runs 2-4 averaged.
Official SGLang JSONL is preserved and validated exactly; a partial request,
token-count mismatch, subprocess error, or missing metric returns non-zero.
EOF
}

ROLE=""
MASTER_ADDR=""
MODEL_SELECTION=all
BUSINESS_IFACE="${BUSINESS_IFACE:-business}"
BASE_API_PORT=30000
BASE_DIST_PORT=21000
BASE_NCCL_PORT=29765
REQUESTED_RESULT_DIR=""
SERVER_PID=""

while (($#)); do
  case "$1" in
    --role)
      (($# >= 2)) || die "--role requires a value"
      ROLE="$2"
      shift 2
      ;;
    --master-addr)
      (($# >= 2)) || die "--master-addr requires a value"
      MASTER_ADDR="$2"
      shift 2
      ;;
    --model)
      (($# >= 2)) || die "--model requires a value"
      MODEL_SELECTION="$2"
      shift 2
      ;;
    --interface)
      (($# >= 2)) || die "--interface requires a value"
      BUSINESS_IFACE="$2"
      shift 2
      ;;
    --port)
      (($# >= 2)) || die "--port requires a value"
      BASE_API_PORT="$2"
      shift 2
      ;;
    --dist-port)
      (($# >= 2)) || die "--dist-port requires a value"
      BASE_DIST_PORT="$2"
      shift 2
      ;;
    --nccl-port)
      (($# >= 2)) || die "--nccl-port requires a value"
      BASE_NCCL_PORT="$2"
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

case "${ROLE}" in
  master|worker) ;;
  *) die "--role must be master or worker" ;;
esac
[[ -n "${MASTER_ADDR}" ]] || die "--master-addr is required"
case "${MODEL_SELECTION}" in
  all|27b|35b) ;;
  *) die "--model must be one of: all, 27b, 35b" ;;
esac
for port_value in "${BASE_API_PORT}" "${BASE_DIST_PORT}" "${BASE_NCCL_PORT}"; do
  [[ "${port_value}" =~ ^[0-9]+$ ]] && ((port_value > 0 && port_value < 65526)) \
    || die "ports must be integers in 1..65525"
done

MODEL_27B_PATH="${MODEL_27B_PATH:-/models/Qwen3.6-27B}"
MODEL_35B_PATH="${MODEL_35B_PATH:-/models/Qwen3.6-35B-A3B}"

require_command "${PYTHON_BIN}"
if [[ "${MODEL_SELECTION}" == all || "${MODEL_SELECTION}" == 27b ]]; then
  require_directory "${MODEL_27B_PATH}"
fi
if [[ "${MODEL_SELECTION}" == all || "${MODEL_SELECTION}" == 35b ]]; then
  require_directory "${MODEL_35B_PATH}"
fi

configure_performance_mode
configure_network "${BUSINESS_IFACE}"
require_npus 4
make_result_dir "${REQUESTED_RESULT_DIR}" "multinode_benchmark_${ROLE}"
write_environment_manifest

stop_master_server() {
  local attempt
  [[ -n "${SERVER_PID}" ]] || return 0
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    log "stopping master server PID ${SERVER_PID}"
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    for ((attempt = 0; attempt < 30; attempt++)); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      log "server did not stop after 30 seconds; sending SIGKILL"
      kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
  fi
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
}
trap stop_master_server EXIT

run_benchmark_model() {
  local label="$1"
  local model_path="$2"
  local offset="$3"
  local api_port=$((BASE_API_PORT + offset))
  local dist_port=$((BASE_DIST_PORT + offset))
  local nccl_port=$((BASE_NCCL_PORT + offset))
  local node_rank=1
  local server_log="${RESULT_DIR}/${label}_${ROLE}_server.log"
  [[ "${ROLE}" == master ]] && node_rank=0
  local -a server_cmd=(
    "${PYTHON_BIN}" -m sglang.launch_server
    --model-path "${model_path}"
    --tp-size 4
    --pp-size 2
    --nnodes 2
    --node-rank "${node_rank}"
    --dist-init-addr "${MASTER_ADDR}:${dist_port}"
    --nccl-port "${nccl_port}"
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

  if [[ "${ROLE}" == master ]]; then
    printf '[ascend-acceptance] command:' | tee -a "${server_log}"
    printf ' %q' "${server_cmd[@]}" | tee -a "${server_log}"
    printf '\n' | tee -a "${server_log}"
    "${server_cmd[@]}" > >(tee -a "${server_log}") 2>&1 &
    SERVER_PID=$!
    log "${label}: master server PID ${SERVER_PID}; waiting for health"
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
    stop_master_server
  else
    log "${label}: worker joining ${MASTER_ADDR}:${dist_port} (TP=4, PP=2)"
    run_logged "${server_log}" \
      "${PYTHON_BIN}" "${SCRIPT_DIR}/run_sglang_worker.py" \
      --expected-schedulers 4 -- "${server_cmd[@]}"
  fi
}

if [[ "${MODEL_SELECTION}" == all || "${MODEL_SELECTION}" == 27b ]]; then
  run_benchmark_model qwen3_6_27b "${MODEL_27B_PATH}" 0
fi
if [[ "${MODEL_SELECTION}" == all || "${MODEL_SELECTION}" == 35b ]]; then
  run_benchmark_model qwen3_6_35b_a3b "${MODEL_35B_PATH}" 10
fi

log "PASS: ${ROLE} completed the requested two-node benchmark matrix"
