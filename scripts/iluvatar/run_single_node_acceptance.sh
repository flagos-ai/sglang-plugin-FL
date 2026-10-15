#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export SGLANG_PLATFORM=sglang_fl
export SGLANG_PLUGINS=sglang_fl
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_IS_FLASHINFER_AVAILABLE=0
export USE_FLAGGEMS=1
export SGLANG_FL_DIST_BACKEND="${SGLANG_FL_DIST_BACKEND:-flagcx}"
export FLAGCX_PATH="${FLAGCX_PATH:-/sgl-workspace/FlagCX}"
export SGLANG_WARMUP_TIMEOUT="${SGLANG_WARMUP_TIMEOUT:-3600}"
export WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-3600}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TP_SIZE="${TP_SIZE:-4}"
export CONCURRENT_N="${CONCURRENT_N:-8}"
export MAX_TOKENS="${MAX_TOKENS:-64}"
export PYTHONUNBUFFERED=1
fused_profile="${FUSED_PROFILE:-vendor}"
if [[ "$fused_profile" == flagos ]]; then
    export SGLANG_FL_PER_OP='gemma_rms_norm=reference;fused_moe=flagos;chunk_gated_delta_rule=flagos;fused_recurrent_gated_delta_rule=flagos;fused_recurrent_gated_delta_rule_packed_decode=flagos'
elif [[ "$fused_profile" != vendor ]]; then
    printf 'FUSED_PROFILE must be vendor or flagos\n' >&2
    exit 2
fi

output_dir="${OUTPUT_DIR:-/tmp/iluvatar-acceptance-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$output_dir"
python scripts/iluvatar/verify_environment.py --min-devices "$TP_SIZE" --fused-profile "$fused_profile" \
    2>&1 | tee "$output_dir/environment.log"
if [[ "$SGLANG_FL_DIST_BACKEND" == flagcx && "$TP_SIZE" -gt 1 ]]; then
    python -m torch.distributed.run --standalone --nproc_per_node="$TP_SIZE" \
        scripts/iluvatar/check_flagcx.py 2>&1 | tee "$output_dir/flagcx.log"
fi

for model in 27b 35b_a3b; do
    if [[ "$model" == 27b ]]; then
        export MODEL_PATH="${DENSE_MODEL_PATH:-/models/Qwen3.6-27B}"
    else
        export MODEL_PATH="${MOE_MODEL_PATH:-/models/Qwen3.6-35B-A3B}"
    fi
    for disabled in 1 0; do
        export DISABLE_CUDA_GRAPH="$disabled"
        python "examples/qwen3_6_${model}_offline_inference.py" \
            2>&1 | tee "$output_dir/${model}_offline_graph_disabled_${disabled}.log"
    done
    python "examples/qwen3_6_${model}_concurrent.py" --mode all \
        2>&1 | tee "$output_dir/${model}_concurrent.log"
done
printf 'Acceptance completed: %s\n' "$output_dir"
