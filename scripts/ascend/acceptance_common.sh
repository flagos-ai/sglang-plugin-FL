#!/usr/bin/env bash
# Shared, strict helpers for Ascend acceptance entrypoints.

set -Eeuo pipefail

ASCEND_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${ASCEND_SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
IMAGE_DIR="${IMAGE_DIR:-${REPO_ROOT}/examples/test_images}"

log() {
  printf '[ascend-acceptance] %s\n' "$*"
}

die() {
  printf '[ascend-acceptance] ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_directory() {
  [[ -d "$1" ]] || die "required directory not found: $1"
}

require_test_images() {
  local image
  local -a required_images=(
    red_square.jpg
    cat.jpg
    stop_sign.png
    digit_seven.png
  )

  require_directory "${IMAGE_DIR}"
  for image in "${required_images[@]}"; do
    [[ -s "${IMAGE_DIR}/${image}" ]] \
      || die "required test image is missing or empty: ${IMAGE_DIR}/${image}"
  done
}

require_npus() {
  local required_count="$1"
  "${PYTHON_BIN}" - "${required_count}" <<'PY'
import sys

import torch

required = int(sys.argv[1])
if not hasattr(torch, "npu") or not torch.npu.is_available():
    raise SystemExit("Ascend torch.npu is unavailable")
count = torch.npu.device_count()
if count < required:
    raise SystemExit(f"need at least {required} visible NPUs, found {count}")
print(f"Ascend preflight: {count} visible NPU(s)")
PY
}

configure_ascend_common() {
  export PYTHONUNBUFFERED=1
  export SGLANG_PLUGINS="${SGLANG_PLUGINS:-sglang_fl}"
  export USE_FLAGGEMS="${USE_FLAGGEMS:-1}"
  export USE_FLAGTUNE="${USE_FLAGTUNE:-0}"
  export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
  # Follow the upstream Ascend runner when the caller has not selected a
  # HCCL port policy.  HOST/NPU ranges must be configured as a pair; an
  # explicit range or legacy base port is job-owned and is never overwritten.
  if [[ -z "${HCCL_HOST_SOCKET_PORT_RANGE+x}" \
    && -z "${HCCL_NPU_SOCKET_PORT_RANGE+x}" \
    && -z "${HCCL_IF_BASE_PORT+x}" ]]; then
    export HCCL_HOST_SOCKET_PORT_RANGE=auto
    export HCCL_NPU_SOCKET_PORT_RANGE=auto
  fi
  export SGLANG_FL_WATCHDOG_DIAG="${SGLANG_FL_WATCHDOG_DIAG:-1}"
  export SGLANG_FL_DIST_BACKEND="${SGLANG_FL_DIST_BACKEND:-flagcx}"
  export FLAGCX_PATH="${FLAGCX_PATH:-/opt/FlagCX}"
  export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}}"
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-${ASCEND_VISIBLE_DEVICES}}"
  export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}"
  export SGLANG_FL_PER_OP="${SGLANG_FL_PER_OP:-silu_and_mul=flagos;mrotary_embedding=flagos;topk=vendor;gemma_rms_norm=vendor;fused_moe=vendor;chunk_gated_delta_rule=vendor}"
  unset ASCEND_LAUNCH_BLOCKING || true
}

configure_correctness_mode() {
  configure_ascend_common
  export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
  export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2400}"
}

configure_performance_mode() {
  configure_ascend_common
  export SGLANG_ENABLE_OVERLAP_PLAN_STREAM="${SGLANG_ENABLE_OVERLAP_PLAN_STREAM:-1}"
  export SGLANG_NPU_USE_MULTI_STREAM="${SGLANG_NPU_USE_MULTI_STREAM:-1}"
  export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
  export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1000}"
  export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
}

configure_network() {
  local interface="$1"
  [[ -n "${interface}" ]] || die "network interface must not be empty"
  export GLOO_SOCKET_IFNAME="${interface}"
  export NCCL_SOCKET_IFNAME="${interface}"
  export HCCL_SOCKET_IFNAME="${interface}"
}

make_result_dir() {
  local requested="$1"
  local suite="$2"
  if [[ -n "${requested}" ]]; then
    RESULT_DIR="${requested}"
  else
    RESULT_DIR="${RESULT_ROOT:-/tmp/sglang-fl-ascend-acceptance}/${suite}_$(date +%Y%m%d_%H%M%S)"
  fi
  mkdir -p -- "${RESULT_DIR}"
  RESULT_DIR="$(cd -- "${RESULT_DIR}" && pwd)"
  export RESULT_DIR
  log "artifacts: ${RESULT_DIR}"
}

run_logged() {
  local log_file="$1"
  shift
  mkdir -p -- "$(dirname -- "${log_file}")"
  printf '[ascend-acceptance] command:' | tee -a "${log_file}"
  printf ' %q' "$@" | tee -a "${log_file}"
  printf '\n' | tee -a "${log_file}"

  set +e
  "$@" 2>&1 | tee -a "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e
  ((status == 0)) || die "command failed with exit code ${status}; see ${log_file}"
}

wait_for_health() {
  local host="$1"
  local port="$2"
  local timeout="$3"
  local server_pid="${4:-}"
  "${PYTHON_BIN}" - "${host}" "${port}" "${timeout}" "${server_pid}" <<'PY'
import os
import sys
import time
import urllib.request

host, port, timeout = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
server_pid = int(sys.argv[4]) if sys.argv[4] else None
url = f"http://{host}:{port}/health"
deadline = time.monotonic() + timeout
last_error = "no response"
while time.monotonic() < deadline:
    if server_pid is not None:
        try:
            os.kill(server_pid, 0)
        except ProcessLookupError:
            raise SystemExit(f"server process {server_pid} exited before health was ready")
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status == 200:
                print(f"server ready: {url}")
                raise SystemExit(0)
            last_error = f"HTTP {response.status}"
    except Exception as exc:
        last_error = str(exc)
    time.sleep(2)
raise SystemExit(f"server did not become healthy in {timeout}s: {url} ({last_error})")
PY
}

write_environment_manifest() {
  local manifest="${RESULT_DIR}/environment.txt"
  {
    printf 'date=%s\n' "$(date --iso-8601=seconds)"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'repo_root=%s\n' "${REPO_ROOT}"
    printf 'python=%s\n' "${PYTHON_BIN}"
    printf 'ASCEND_VISIBLE_DEVICES=%s\n' "${ASCEND_VISIBLE_DEVICES}"
    printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "${ASCEND_RT_VISIBLE_DEVICES}"
    printf 'SGLANG_PLUGINS=%s\n' "${SGLANG_PLUGINS}"
    printf 'SGLANG_FL_DIST_BACKEND=%s\n' "${SGLANG_FL_DIST_BACKEND}"
    printf 'SGLANG_FL_PER_OP=%s\n' "${SGLANG_FL_PER_OP}"
    printf 'FLAGCX_PATH=%s\n' "${FLAGCX_PATH}"
    printf 'SGLANG_ENABLE_OVERLAP_PLAN_STREAM=%s\n' "${SGLANG_ENABLE_OVERLAP_PLAN_STREAM}"
    printf 'SGLANG_NPU_USE_MULTI_STREAM=%s\n' "${SGLANG_NPU_USE_MULTI_STREAM:-<unset>}"
    printf 'STREAMS_PER_DEVICE=%s\n' "${STREAMS_PER_DEVICE:-<unset>}"
    printf 'HCCL_BUFFSIZE=%s\n' "${HCCL_BUFFSIZE:-<unset>}"
    printf 'HCCL_OP_EXPANSION_MODE=%s\n' "${HCCL_OP_EXPANSION_MODE:-<unset>}"
    printf 'HCCL_SOCKET_IFNAME=%s\n' "${HCCL_SOCKET_IFNAME:-<unset>}"
    printf 'GLOO_SOCKET_IFNAME=%s\n' "${GLOO_SOCKET_IFNAME:-<unset>}"
    printf 'HCCL_HOST_SOCKET_PORT_RANGE=%s\n' "${HCCL_HOST_SOCKET_PORT_RANGE:-<unset>}"
    printf 'HCCL_NPU_SOCKET_PORT_RANGE=%s\n' "${HCCL_NPU_SOCKET_PORT_RANGE:-<unset>}"
    printf 'HCCL_IF_BASE_PORT=%s\n' "${HCCL_IF_BASE_PORT:-<unset>}"
    printf 'NCCL_SOCKET_IFNAME=%s\n' "${NCCL_SOCKET_IFNAME:-<unset>}"
    "${PYTHON_BIN}" -V
    "${PYTHON_BIN}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

for package in ("sglang", "sglang-fl", "torch", "torch-npu"):
    try:
        print(f"{package}={version(package)}")
    except PackageNotFoundError:
        print(f"{package}=NOT_INSTALLED")
PY
  } >"${manifest}"
}
