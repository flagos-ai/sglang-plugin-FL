#!/usr/bin/env bash
# Run separately on an eight-H100 prefill or decode node. Install SGLang with
# the V4.1 model/operator backports and this plugin's dsv41 extra first.
# Required: MODEL_PATH, ROLE=prefill|decode, HOST_IP, IB_DEVICES, FLAGCX_PATH.
# Configure FLAGCX_SOCKET_IFNAME and the FlagCX/RDMA environment for your fabric.
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to a DeepSeek V4.1 Flash checkpoint}"
: "${ROLE:?Set ROLE to prefill or decode}"
: "${HOST_IP:?Set HOST_IP to a routable node address}"
: "${IB_DEVICES:?Set IB_DEVICES to the comma-separated RDMA interfaces}"
: "${FLAGCX_PATH:?Set FLAGCX_PATH to the FlagCX installation}"
export FLAGCX_PATH SGLANG_HOST_IP="$HOST_IP"
export SGLANG_PLUGINS="${SGLANG_PLUGINS:-sglang_fl,sglang_dsv41,flaggems_dsv41}"
args=(
  --model-path "$MODEL_PATH" --served-model-name deepseek-v4.1-flash
  --trust-remote-code --tp 8 --ep-size 8 --context-length 1048576
  --host 0.0.0.0 --port "${PORT:-31000}"
  --disaggregation-mode "$ROLE" --disaggregation-transfer-backend flagcx
  --disaggregation-ib-device "$IB_DEVICES" --load-balance-method follow_bootstrap_room
  --moe-a2a-backend none --speculative-algorithm DSPARK
  --speculative-dspark-block-size 5 --mm-enable-dp-encoder
)
case "$ROLE" in
  prefill)
    args+=(--disaggregation-bootstrap-port "${BOOTSTRAP_PORT:-9010}"
           --chunked-prefill-size 8192 --prefill-max-requests 4 --disable-cuda-graph)
    ;;
  decode)
    args+=(--dp-size 4 --enable-dp-attention --enable-dp-lm-head
           --dsv41-encoder-only-prefill --cuda-graph-max-bs-decode 96)
    ;;
  *) echo "ROLE must be prefill or decode" >&2; exit 2 ;;
esac
exec python -m sglang.launch_server "${args[@]}" "$@"
