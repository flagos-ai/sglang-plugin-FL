#!/bin/bash

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================================
# SGLang 验收脚本: Original SGLang vs sglang-plugin-FL
# ============================================================
#
# 使用方式:
#   MODEL_PATH=/path/to/model bash validate.sh all
#   MODEL_PATH=/path/to/model bash validate.sh full   # 含 vendor 验证
#
#   TP_SIZE=8 MODEL_PATH=/path/to/model \
#     bash validate.sh all
#
# 输出文件（tp=1）:
#   /tmp/dispatch_baseline.log
#   /tmp/dispatch_plugin.log
#   /tmp/dispatch_vendor.log
#   /tmp/gems_aten.txt
#   <tests_dir>/precision_results.json
#
# 输出文件（tp>1，如 tp=8）:
#   /tmp/dispatch_baseline_tp8.log
#   /tmp/dispatch_plugin_tp8.log
#   /tmp/dispatch_vendor_tp8.log
#   /tmp/gems_aten_tp8.txt
#   <tests_dir>/precision_results_tp8.json
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

# --- 参数 ---
TP_SIZE=${TP_SIZE:-1}
MODEL_PATH=${MODEL_PATH:-""}
if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH environment variable is required."
    echo "  Example: MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct bash validate.sh all"
    exit 1
fi
# No PYTHONPATH needed — plugin auto-discovered via setuptools entry_points

# 日志文件后缀（tp>1 时加 _tp{N} 后缀，避免覆盖 tp=1 的结果）
if [ "$TP_SIZE" -gt 1 ]; then
    SUFFIX="_tp${TP_SIZE}"
else
    SUFFIX=""
fi

DISPATCH_BASELINE=/tmp/dispatch_baseline${SUFFIX}.log
DISPATCH_PLUGIN=/tmp/dispatch_plugin${SUFFIX}.log
DISPATCH_VENDOR=/tmp/dispatch_vendor${SUFFIX}.log
GEMS_ATEN=/tmp/gems_aten${SUFFIX}.txt
RESULT_FILE=${SCRIPT_DIR}/precision_results${SUFFIX}.json

LOG_FILES=(
    "$DISPATCH_BASELINE"
    "$DISPATCH_PLUGIN"
    "$DISPATCH_VENDOR"
    "$GEMS_ATEN"
    "$RESULT_FILE"
)

# --- 子命令 ---
clean_logs() {
    echo "--- Cleaning previous logs (tp=${TP_SIZE}) ---"
    rm -f "${LOG_FILES[@]}"
}

run_baseline() {
    echo "============================================================"
    echo "  Step 1: Original SGLang (no plugin, tp=${TP_SIZE})"
    echo "============================================================"
    rm -f "$DISPATCH_BASELINE"
    touch "$DISPATCH_BASELINE"

    SGLANG_PLUGINS="__none__" \
    SGLANG_FL_DISPATCH_LOG="$DISPATCH_BASELINE" \
    MODEL_PATH="$MODEL_PATH" \
    TP_SIZE="$TP_SIZE" \
        python test_precision_align.py baseline

    echo ""
    echo "--- Baseline Dispatch Log (unique ops) ---"
    sort -u "$DISPATCH_BASELINE"
    echo ""
}

run_plugin() {
    echo "============================================================"
    echo "  Step 2: sglang-plugin-FL (FlagGems + OOT dispatch, tp=${TP_SIZE})"
    echo "============================================================"
    rm -f "$DISPATCH_PLUGIN" "$GEMS_ATEN"

    SGLANG_FL_DISPATCH_LOG="$DISPATCH_PLUGIN" \
    SGLANG_FLAGGEMS_RECORD=1 \
    SGLANG_FLAGGEMS_LOG_ONCE=1 \
    SGLANG_FLAGGEMS_LOG_PATH="$GEMS_ATEN" \
    MODEL_PATH="$MODEL_PATH" \
    TP_SIZE="$TP_SIZE" \
        python test_precision_align.py plugin

    echo ""
    echo "--- Plugin Dispatch Log (unique ops) ---"
    sort -u "$DISPATCH_PLUGIN"
    echo ""
    echo "--- FlagGems ATen Replacement Log ---"
    cat "$GEMS_ATEN"
    echo ""
}

run_vendor() {
    echo "============================================================"
    echo "  Step 2b: Mock Vendor dispatch (mock_npu, tp=${TP_SIZE})"
    echo "============================================================"
    rm -f "$DISPATCH_VENDOR"

    SGLANG_FL_DISPATCH_LOG="$DISPATCH_VENDOR" \
    SGLANG_FL_PER_OP="silu_and_mul=vendor:mock_npu;rms_norm=vendor:mock_npu" \
    SGLANG_MOCK_NPU_AVAILABLE=1 \
    SGLANG_MOCK_NPU_LOG="$DISPATCH_VENDOR" \
    SGLANG_FLAGGEMS_RECORD=1 \
    SGLANG_FLAGGEMS_LOG_ONCE=1 \
    SGLANG_FLAGGEMS_LOG_PATH="$GEMS_ATEN" \
    MODEL_PATH="$MODEL_PATH" \
    TP_SIZE="$TP_SIZE" \
        python test_precision_align.py plugin

    echo ""
    echo "--- Vendor Dispatch + Runtime Log ---"
    cat "$DISPATCH_VENDOR"
    echo ""
    echo "Expected: SiluAndMul/RMSNorm → vendor(vendor.mock_npu) + MOCK_NPU_RUNTIME entries"
    echo ""
}

run_compare() {
    echo "============================================================"
    echo "  Step 3: Precision Compare (baseline vs plugin, tp=${TP_SIZE})"
    echo "============================================================"
    MODEL_PATH="$MODEL_PATH" \
    TP_SIZE="$TP_SIZE" \
        python test_precision_align.py compare

    echo ""
    echo "============================================================"
    echo "  Dispatch Diff: baseline vs plugin"
    echo "============================================================"
    echo "Baseline (unique):" && sort -u "$DISPATCH_BASELINE"
    echo ""
    echo "Plugin (unique):"   && sort -u "$DISPATCH_PLUGIN"
    echo ""
    echo "Diff:"
    diff <(sort -u "$DISPATCH_BASELINE") <(sort -u "$DISPATCH_PLUGIN") || true

    echo ""
    echo "============================================================"
    echo "  FlagGems ATen Ops Replaced (plugin mode only)"
    echo "============================================================"
    echo "Total ATen ops logged: $(wc -l < "$GEMS_ATEN")"
    cat "$GEMS_ATEN"

    echo ""
    echo "============================================================"
    echo "  Summary (tp=${TP_SIZE})"
    echo "============================================================"
    echo "  Model:               $MODEL_PATH"
    echo "  Baseline dispatch:   $DISPATCH_BASELINE"
    echo "  Plugin dispatch:     $DISPATCH_PLUGIN"
    echo "  FlagGems ATen log:   $GEMS_ATEN"
    echo "  Precision results:   $RESULT_FILE"
}

case "${1:-help}" in
    clean)    clean_logs ;;
    baseline) run_baseline ;;
    plugin)   run_plugin ;;
    vendor)   run_vendor ;;
    compare)  run_compare ;;
    all)
        clean_logs
        run_baseline
        run_plugin
        run_compare
        ;;
    full)
        clean_logs
        run_baseline
        run_plugin
        run_vendor
        run_compare
        ;;
    *)
        echo "Usage: MODEL_PATH=... [TP_SIZE=N] bash validate.sh {clean|baseline|plugin|vendor|compare|all|full}"
        echo ""
        echo "  clean     — Remove all previous log files"
        echo "  baseline  — Run original SGLang (no plugin)"
        echo "  plugin    — Run with sglang-plugin-FL (flagos backend)"
        echo "  vendor    — Run with mock_npu vendor backend"
        echo "  compare   — Compare precision + dispatch diff + ATen summary"
        echo "  all       — clean + baseline + plugin + compare"
        echo "  full      — clean + baseline + plugin + vendor + compare"
        echo ""
        echo "Examples:"
        echo "  MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct bash validate.sh all"
        echo "  TP_SIZE=8 MODEL_PATH=/path/to/14B bash validate.sh full"
        exit 1
        ;;
esac
