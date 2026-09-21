#!/usr/bin/env bash
# Single-node Ascend correctness suite for the SGLang 0.5.18 adaptation.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=acceptance_common.sh
source "${SCRIPT_DIR}/acceptance_common.sh"

usage() {
  cat <<'EOF'
Usage: bash scripts/ascend/run_single_node_acceptance.sh [options]

Runs, by default, the complete single-node acceptance matrix:
  * Qwen3.6-27B and Qwen3.6-35B-A3B offline validation at TP=4
  * both concurrent suites (text/VL/mixed) at TP=4
  * three independent text-concurrency canary runs at TP=2 for each model
  * Qwen3.6-27B MTP-vs-baseline validation at TP=4

Options:
  --model {all|27b|35b}  Limit the model set (default: all).
  --result-dir PATH      Artifact directory (default: timestamp under /tmp).
  -h, --help             Show this help.

Environment:
  MODEL_27B_PATH         Default: /models/Qwen3.6-27B
  MODEL_35B_PATH         Default: /models/Qwen3.6-35B-A3B
  IMAGE_DIR              Default: examples/test_images
  PYTHON_BIN             Default: python3
  CONCURRENT_N           Concurrent-example fanout (default: script default).
EOF
}

MODEL_SELECTION=all
REQUESTED_RESULT_DIR=""
while (($#)); do
  case "$1" in
    --model)
      (($# >= 2)) || die "--model requires a value"
      MODEL_SELECTION="$2"
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
    *)
      die "unknown argument: $1"
      ;;
  esac
done

case "${MODEL_SELECTION}" in
  all|27b|35b) ;;
  *) die "--model must be one of: all, 27b, 35b" ;;
esac

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
require_npus 4
make_result_dir "${REQUESTED_RESULT_DIR}" single_node
write_environment_manifest

run_model_suite() {
  local label="$1"
  local model_path="$2"
  local offline_script="$3"
  local concurrent_script="$4"
  local canary_run

  log "${label}: TP=4 offline correctness"
  run_logged "${RESULT_DIR}/${label}_tp4_offline.log" \
    env MODEL_PATH="${model_path}" TP_SIZE=4 IMAGE_DIR="${IMAGE_DIR}" \
    "${PYTHON_BIN}" "${REPO_ROOT}/examples/${offline_script}"

  log "${label}: TP=4 concurrent text/VL/mixed correctness"
  run_logged "${RESULT_DIR}/${label}_tp4_concurrent.log" \
    env MODEL_PATH="${model_path}" TP_SIZE=4 IMAGE_DIR="${IMAGE_DIR}" \
    "${PYTHON_BIN}" "${REPO_ROOT}/examples/${concurrent_script}" --mode all

  for canary_run in 1 2 3; do
    log "${label}: TP=2 concurrent canary ${canary_run}/3"
    run_logged "${RESULT_DIR}/${label}_tp2_canary_${canary_run}.log" \
      env MODEL_PATH="${model_path}" TP_SIZE=2 IMAGE_DIR="${IMAGE_DIR}" \
      "${PYTHON_BIN}" "${REPO_ROOT}/examples/${concurrent_script}" --mode text
  done
}

if [[ "${MODEL_SELECTION}" == all || "${MODEL_SELECTION}" == 27b ]]; then
  run_model_suite \
    qwen3_6_27b \
    "${MODEL_27B_PATH}" \
    qwen3_6_27b_offline_inference.py \
    qwen3_6_27b_concurrent.py

  log "qwen3_6_27b: TP=4 MTP-vs-baseline correctness"
  run_logged "${RESULT_DIR}/qwen3_6_27b_tp4_mtp.log" \
    env MODEL_PATH="${MODEL_27B_PATH}" TP_SIZE=4 \
    "${PYTHON_BIN}" "${REPO_ROOT}/examples/qwen3_6_27b_mtp_inference.py" \
    --disable-cuda-graph --disable-piecewise-cuda-graph --disable-overlap-schedule
fi

if [[ "${MODEL_SELECTION}" == all || "${MODEL_SELECTION}" == 35b ]]; then
  run_model_suite \
    qwen3_6_35b_a3b \
    "${MODEL_35B_PATH}" \
    qwen3_6_35b_a3b_offline_inference.py \
    qwen3_6_35b_a3b_concurrent.py
fi

log "PASS: single-node acceptance completed; no command was skipped"
