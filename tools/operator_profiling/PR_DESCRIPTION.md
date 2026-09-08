# feat: add eager operator inventory profiling for SGLang workloads

## 背景

算子开发需要知道模型在 SGLang 中实际执行了哪些计算：上层算子 API、设备 kernel、输入
shape/dtype、执行次数和累计设备时间。只枚举 Python 或 ATen API 无法覆盖 SGLang/sgl-kernel、
FlashInfer、plugin vendor dispatch 以及框架内部局部编译产生的 kernel；直接交付 profiler
trace 又体积大、难以聚合，也缺少来源和守恒校验。

本 PR 在 sglang-plugin-FL 中增加一套基于 `torch.profiler` 的 eager 算子清单采集工具，将
真实 TP 推理 trace 归一化为算子团队可以直接使用和独立验收的三文件物料。

## 采集口径

采集使用新增的 `platform_profile` 模式：

- 关闭 FlagGems 对 ATen API 的替换；
- 不选择当前 FlagOS fused 实现；
- 保留 sglang-plugin-FL vendor dispatch；
- vendor adapter 不可用时保留 SGLang 为当前平台选择的原生实现；
- 使用 eager 执行，显式关闭全局 Torch Compile 和 CUDA Graph；
- 框架内部实际执行的局部 TorchInductor kernel 仍会识别为 `torch_fused`；
- warmup 在 profiler 启动前完成；
- 通信、GPU memcpy 和 GPU memset 不进入计算算子清单。

计算 kernel 按实际运行路径归为 `torch_aten`、`torch_fused`、`third_party` 和 `vendor`
四类。`vendor` 必须有实际进入 plugin `VENDOR OpImpl` 的运行时 marker，不能依据 CUDA 或
kernel 名称猜测。

当前 PR 只交付准确的算子清单，不推断尚未接入的 FlagGems-sglang 与现有融合算子的功能
等价关系，也不计算融合算子覆盖率。相关接入和映射列在 README 的 TODO 中。

本 PR 以已合入 [PR #69](https://github.com/flagos-ai/sglang-plugin-FL/pull/69)
strict 语义修复的主分支为基线：`strict=True` 表示首选实现失败时立即报错，不执行实现
fallback。

## 主要改动

- 增加 `SGLANG_FL_MODE=platform_profile`，并保持默认 `adapt` 行为不变；
- 将 CLI、环境检查、trace 解析、报表生成、文档和可提交样例统一收敛到
  `tools/operator_profiling/`；plugin 侧只保留必须进入 wheel 的
  `sglang_fl/profiling_hooks.py`；
- 增加采集前自动环境门禁，检查模型、GPU、entry point、实际源码路径、FlagTree provider、
  SGLang Engine 真实导入和可选的严格版本基线；
- 为 SGLang JIT、Triton/FlagTree callable、MultiPlatformOp 和 plugin vendor dispatch 增加
  仅在 profiler 活跃时生效的 correlation marker；
- 从各 TP rank 的 torch profiler trace 关联 API、kernel、shape/dtype 和 dispatch 证据；
- 稳定化动态 kernel 名，区分 eager 与框架内部局部 compile；
- 跨 rank 合并名称/shape，并累计 kernel 执行次数与设备 duration；
- 生成固定三文件物料并提供独立落盘校验器；
- 对齐本次 H20 验证实际使用的 sgl-kernel 0.5.11 CUDA adapter 调用约定；
- 增加字段说明、最小验证命令、正式采集命令和物料阅读说明。

每个 workload 的正式输出恰好为：

```text
results/
├── operator_list.csv
├── kernel_details_report.csv
└── summary.json
```

原始 trace 和归一化审计表保存在运行目录，用于问题追溯，不提交源码仓库。

仓库内同时提交一份约 58 KB 的真实 Qwen3.6-27B TP4 smoke 样例：

```text
tools/operator_profiling/examples/qwen3_6_27b_tp4_smoke/
├── README.md
└── results/
    ├── operator_list.csv
    ├── kernel_details_report.csv
    └── summary.json
```

样例用于 Review schema 和校验流程；正式 H20 物料与原始 trace 不进入源码 PR。

## 使用方法

在仓库根目录先运行一个 TP4 小型真实模型验证：

```bash
python -m pip install -e . --no-deps

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python tools/operator_profiling/check_environment.py \
  --model-path /models/Qwen3.6-27B \
  --tp-size 4 \
  --require-validated-versions

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -u tools/operator_profiling/run_operator_profile.py \
  --model-path /models/Qwen3.6-27B \
  --tp-size 4 \
  --input-tokens 16 \
  --output-tokens 4 \
  --concurrency 2 \
  --mem-fraction-static 0.80 \
  --chunked-prefill-size 8192 \
  --attention-backend triton \
  --require-validated-versions \
  --output-dir tools/operator_profiling/runs/smoke
```

成功标志为：

```text
EAGER_OPERATOR_PROFILE_PASS output=<run-directory>
```

落盘后可独立复核任意 workload：

```bash
python tools/operator_profiling/validate_operator_inventory.py \
  <run-directory>/<workload>/results
```

成功标志为 `OPERATOR_INVENTORY_VALIDATION_PASS`。正式 27B/35B 采集命令和所有字段定义见
`tools/operator_profiling/README.md` 与 `tools/operator_profiling/MATERIAL_GUIDE.md`。

## 已验证环境

| 项目 | 版本或配置 |
| --- | --- |
| GPU | NVIDIA H20-3e × 4 |
| SGLang | v0.5.11 tag（`612785ffdcaf`） |
| sglang-plugin-FL 目标基线 | `main@0f63052`，包含 PR #69 |
| PyTorch | 2.11.0+cu130 |
| FlagTree | 0.6.2a1 |
| 执行模式 | eager、TP4、全局 Torch Compile 关闭、CUDA Graph 关闭 |

## 已验证模型与物料

以下六个正式 workload 均生成三文件物料，覆盖 ranks `[0,1,2,3]`，并通过源报告守恒检查
和独立落盘校验：

| 模型 | 输入 / 输出 | 并发 | all-ranks 计算 kernel 执行次数 | 结果 |
| --- | ---: | ---: | ---: | --- |
| Qwen3.6-27B | 1K / 1K | 64 | 4,947,436 | PASS |
| Qwen3.6-27B | 4K / 1K | 64 | 5,112,460 | PASS |
| Qwen3.6-27B | 16K / 1K | 64 | 5,769,740 | PASS |
| Qwen3.6-35B-A3B | 1K / 1K | 64 | 4,291,532 | PASS |
| Qwen3.6-35B-A3B | 4K / 1K | 64 | 4,431,116 | PASS |
| Qwen3.6-35B-A3B | 16K / 1K | 64 | 4,986,636 | PASS |

正式物料批次标识：`platform_profile_h20_20260904`。当前收口代码在 PR #69 基线上的 TP4
smoke 批次标识：`platform_profile_h20_20260908_pr69_smoke`。两批本地物料均保存在
`tools/operator_profiling/materials/`，并由 `.gitignore` 排除；制品不进入本 PR。对外分发时保留目录
结构以及 `summary.json` 记录的 CSV 行数和 SHA-256。256K 尚未执行，不属于本 PR 的已验证
范围。

当前收口代码另完成一次 Qwen3.6-27B TP4 smoke 回归：

- 4 ranks 全部采集并合并；
- 实际 dispatch kind 仅为 `vendor`，实现 ID 为 `vendor.cuda`；
- PR #69 strict 语义生效，dispatch 日志为 `mode=direct`；
- 25,280 次计算 kernel 执行全部守恒；
- 对外目录只包含三份正式文件；
- 独立校验器 22 项检查全部通过。

## 验证

```text
dispatch/platform/operator-profiling pytest on upstream main: 217 passed
ruff check (PR files): PASS
ruff format --check (PR Python files): PASS
git diff --check: PASS
wheel build and package-content check: PASS
formal artifact validation: 6/6 workloads PASS, 22 checks each
Qwen3.6-27B TP4 live smoke on upstream main with SGLang v0.5.11: PASS, 22 checks
```

## Review 建议

建议按以下顺序 review：

1. `tools/operator_profiling/README.md`：需求边界、运行口径和使用方式；
2. `sglang_fl/mode.py`、`sglang_fl/__init__.py`、`sglang_fl/dispatch/manager.py`：
   `platform_profile` 是否与默认 `adapt` 隔离；
3. `sglang_fl/profiling_hooks.py`：运行时 marker 是否透明；
4. `tools/operator_profiling/trace_report.py`：trace 关联、来源分类和跨 rank 聚合；
5. `tools/operator_profiling/inventory_report.py`：三文件 schema 和守恒校验；
6. `tools/operator_profiling/MATERIAL_GUIDE.md`：字段定义是否满足算子团队使用需求；
7. `tests/unit_tests/tools/operator_profiling/` 与
   `tests/unit_tests/platform/test_profiling_hooks.py`：边界条件和报表验收规则。
