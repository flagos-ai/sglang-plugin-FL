#!/usr/bin/env bash
# Two-node TP=4, PP=2 correctness suite. Run once on each node.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=acceptance_common.sh
source "${SCRIPT_DIR}/acceptance_common.sh"

usage() {
  cat <<'EOF'
Usage: bash scripts/ascend/run_multinode_examples.sh OPTIONS

Required:
  --role {master|worker}  This node's role/rank (master=0, worker=1).
  --master-addr ADDRESS   Routable address of the master node.

Options:
  --model {all|27b|35b}  Limit the model set (default: all).
  --interface NAME        HCCL/Gloo interface (default: business).
  --port PORT             First API port (default: 30000).
  --dist-port PORT        First rendezvous port (default: 20000).
  --nccl-port PORT        First collective port (default: 28765).
  --result-dir PATH       Local artifact directory (default: /tmp timestamp).
  -h, --help              Show this help.

Run the same model selection on both nodes. The second model uses each base
port plus 10. Each node must expose four NPUs. Images are mandatory on both
nodes; any missing/empty image or non-zero child exit fails the suite.
EOF
}

ROLE=""
MASTER_ADDR=""
MODEL_SELECTION=all
BUSINESS_IFACE="${BUSINESS_IFACE:-business}"
BASE_API_PORT=30000
BASE_DIST_PORT=20000
BASE_NCCL_PORT=28765
REQUESTED_RESULT_DIR=""

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
require_test_images
if [[ "${MODEL_SELECTION}" == all || "${MODEL_SELECTION}" == 27b ]]; then
  require_directory "${MODEL_27B_PATH}"
fi
if [[ "${MODEL_SELECTION}" == all || "${MODEL_SELECTION}" == 35b ]]; then
  require_directory "${MODEL_35B_PATH}"
fi

configure_correctness_mode
configure_network "${BUSINESS_IFACE}"
require_npus 4
make_result_dir "${REQUESTED_RESULT_DIR}" "multinode_examples_${ROLE}"
write_environment_manifest

NODE_RANK=1
[[ "${ROLE}" == master ]] && NODE_RANK=0

run_example() {
  local label="$1"
  local model_path="$2"
  local example_script="$3"
  local offset="$4"
  local api_port=$((BASE_API_PORT + offset))
  local dist_port=$((BASE_DIST_PORT + offset))
  local nccl_port=$((BASE_NCCL_PORT + offset))

  log "${label}: ${ROLE}, TP=4, PP=2, node_rank=${NODE_RANK}"
  run_logged "${RESULT_DIR}/${label}_${ROLE}.log" \
    env MODEL_PATH="${model_path}" IMAGE_DIR="${IMAGE_DIR}" ATTENTION_BACKEND=ascend \
    "${PYTHON_BIN}" "${REPO_ROOT}/examples/${example_script}" \
    --role "${ROLE}" \
    --node-rank "${NODE_RANK}" \
    --master-addr "${MASTER_ADDR}" \
    --tp 4 \
    --pp 2 \
    --nnodes 2 \
    --port "${api_port}" \
    --dist-port "${dist_port}" \
    --nccl-port "${nccl_port}" \
    --max-wait 1200 \
    --request-timeout 600 \
    --text-concurrency 32 \
    --vl-concurrency 8
}

if [[ "${MODEL_SELECTION}" == all || "${MODEL_SELECTION}" == 27b ]]; then
  run_example qwen3_6_27b "${MODEL_27B_PATH}" qwen3_6_27b_multinode.py 0
fi
if [[ "${MODEL_SELECTION}" == all || "${MODEL_SELECTION}" == 35b ]]; then
  run_example qwen3_6_35b_a3b "${MODEL_35B_PATH}" qwen3_6_35b_a3b_multinode.py 10
fi

log "PASS: ${ROLE} completed the requested two-node TP=4, PP=2 example matrix"
