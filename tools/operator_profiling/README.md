# SGLang 运行时算子清单采集

本目录用于采集一次真实 SGLang 推理中执行的算子 API、设备 kernel、输入 shape/dtype、
执行次数和设备时间，并支持原生/插件 Graph 场景的 FlagOS 算子覆盖率。

| `--implementation` | 用途 | 默认执行模式与负载 |
|---|---|---|
| `native` | 原生基线，禁用插件发现，要求独立 stock Triton 环境 | Graph，输入4096/输出256/并发64 |
| `plugin` | 插件采集，记录 FlagGems/FlagTree/FlagCX 运行证据 | Graph，输入4096/输出256/并发64 |
| `platform`（默认） | 原有开发清单，vendor调度、全rank计算kernel汇总 | eager，输入1K/4K/16K、输出1K、并发64 |

## FlagOS 覆盖率工作流

基于本 PR 的 trace 解析器，参考 [vLLM #422](https://github.com/flagos-ai/vllm-plugin-FL/pull/422)
的采集/覆盖率流程及 [#593](https://github.com/flagos-ai/vllm-plugin-FL/pull/593) 的 eager 补全。

- native/plugin 导出全部 TP rank，默认解析 `--rank 0`；纳入计算、通信及未知kernel，排除warmup、Graph capture、memcpy/memset活动。
- eager参考按完整kernel名精确匹配，要求同实现、模型、负载、引擎配置、软件、硬件、provider及rank；歧义候选保留，跳过自动生成的Inductor融合kernel。
- FlagGems/FlagTree要求成功运行证据与实际kernel相交；FlagCX还要求活跃调用关联真实通信kernel。开关或参考信息本身不能证明接管。
- 覆盖率按唯一基线身份计算：`(有运行证据的FlagOS身份 ∩ 基线) ∪ (已核实融合目标 ∩ 基线)` 的去重数量 / 基线数量。未知项保留分母；次数、耗时不作为权重。
- `--denominator` 默认A（原生基线），支持D（同平台原生）和显式B回退；B须填写 `--fallback-reason`。`--fusion-mappings` 接受带源码依据的显式映射，验证实测kernel及基线交集。

### 采集与计算

在已准备好依赖的环境运行。native使用独立stock Triton解释器；plugin环境需通过
`python -m pip install -e . --no-deps` 注册当前checkout。独立SGLang源码可通过 `PYTHONPATH` 引入。
下面分别以 `IMPL=native`、`IMPL=plugin` 及对应解释器执行；每轮先采eager，再采Graph。
两款已验证模型共用此流程，选择本机对应的模型目录：

| 示例 | `MODEL` |
|---|---|
| Qwen3.6-27B | `/models/Qwen3.6-27B` |
| Qwen3.6-35B-A3B | `/models/Qwen3.6-35B-A3B` |

```bash
export PYTHONPATH=/path/to/sglang-plugin-FL:/path/to/sglang/python
export CUDA_VISIBLE_DEVICES=0,1 USE_FLAGTUNE=0 SGLANG_FL_DISAGG_FLAGCX=0
# 本轮Qwen验证使用的排除项；不修改插件默认策略。
export SGLANG_FL_FLAGOS_BLACKLIST=count_nonzero,mm,linear
PYTHON=/path/to/native-env/bin/python  # plugin改为其环境的python
IMPL=native                          # 第二轮改为plugin
MODEL=/models/Qwen3.6-27B             # 35B改为/models/Qwen3.6-35B-A3B
MODEL_NAME=${MODEL##*/}
COMMON=(--implementation "$IMPL" --model-path "$MODEL" --tp-size 2
        --input-tokens 4096 --output-tokens 256 --concurrency 64
        --context-length 8192 --cuda-graph-max-bs 64 --length-overflow-policy error)
"$PYTHON" tools/operator_profiling/run_operator_profile.py "${COMMON[@]}" \
  --execution-mode eager --output-dir "tools/operator_profiling/runs/${MODEL_NAME}_${IMPL}_eager"
# 填入上一步 RUNTIME_OPERATOR_PROFILE_PASS 打印的 results 路径。
EAGER_RESULTS=/path/to/eager/RUN/CASE/results
"$PYTHON" tools/operator_profiling/run_operator_profile.py "${COMMON[@]}" \
  --execution-mode graph --eager-reference "$EAGER_RESULTS" \
  --output-dir "tools/operator_profiling/runs/${MODEL_NAME}_${IMPL}_graph"
```

两轮完成后，传入对应Graph采集的 `results/` 目录：

```bash
python tools/operator_profiling/generate_flagos_coverage.py \
  --baseline /path/to/native_graph/results --plugin /path/to/plugin_graph/results \
  --output-dir "tools/operator_profiling/runs/${MODEL_NAME}_coverage"
```

若只需原始Graph清单，可省略 `--eager-reference`。35B使用相同流程，替换模型路径。
离线重建使用 `runtime_report.py --trace-dir ... --metadata workload.json --output-dir ...`
及可选 `--eager-reference ... --evidence-dir ...`；要保留FlagOS证据，需提供原始evidence目录。

### 输出与校验

| 文件 | 内容 |
|---|---|
| `operator_list.csv` | 5列：ID、API、kind、完整kernel名、时间占比 |
| `kernel_shape_dtype.csv` | 固定14列：归因、shape/dtype、候选、次数/时间及5个eager参考字段；无参考填 `null` |
| `kernel_shape_dtype_audit.csv` | 转换前明细，保留原始dtype/status及source/dispatch；eager参考优先读取此表 |
| `kernel_time.csv`、`operator_normalization.csv` | kernel耗时、归一化身份和规则 |
| `runtime_evidence.json`、`summary.json` | 运行证据、环境/rank/window、逐kernel守恒、参考来源及SHA-256 |
| `operator_flagos_coverage.csv`、`operator_flagos_coverage_summary.csv` | 基线逐项证据、唯一分子/分母、待确认数及覆盖率 |
| `flagos_operators.csv`、`fusion_mapping.csv`、`coverage_summary.json` | FlagOS身份、融合审计及输入摘要 |

两张公共CSV的列名、顺序及字段约定对齐
[指定op_profile示例](https://github.com/cyber-pioneer/op_profile/tree/main/qwen3.6_35b_a3b_plugin_graph_4096_256/results)。
导出层统一kind分类、去除 `native_marker_` 状态前缀，并将dtype转换为profiler写法
（如 `torch.bfloat16→c10::BFloat16`、`Scalar[int]→Scalar`）；原值留audit，身份和计数不变。
新报告为 `flagos-runtime-v2`，校验器兼容v1。

采集及覆盖率计算自动校验SHA-256、逐kernel次数/纳秒守恒和主表/audit一致性。
独立复核使用 `runtime_report.validate_runtime_report(Path(results))`；下文的
`validate_operator_inventory.py` 仅校验旧platform格式。

### 验证记录

H20-3e、TP2、SGLang0.5.11、PyTorch2.11.0+cu130；native为stock Triton3.6.0，
plugin为FlagTree0.6.2a1/FlagGems5.0.2。输入4096/输出256/并发64，
两模型的native/plugin × eager/Graph共8组真实采集通过。

| 模型 | 原始Graph覆盖率 | Graph + eager参考 |
|---|---|---|
| Qwen3.6-27B | 33/63 = 52.38% | 33/50 = 66.00% |
| Qwen3.6-35B-A3B | 36/74 = 48.65% | 36/59 = 61.02% |

2026-10-22：256项测试通过；复用原始trace重建12份报告，原始明细无损，身份、次数、耗时及4组覆盖率不变。
验证排除 `count_nonzero,mm,linear`；原生/插件输出摘要不同，本次结果不证明精度等价或性能收益。
FlagCX仅有针对性测试，本轮实测通信为NCCL；未使用人工融合映射。完整运行材料独立保存，不纳入源码。

## Platform 开发清单

以下说明 `--implementation platform` 路径；其全rank、compute-only清单不可作为覆盖率基线A。
字段解释见 [MATERIAL_GUIDE.md](MATERIAL_GUIDE.md)。

仓库目录按职责拆分：

- `tools/operator_profiling/`：命令行入口、环境检查、trace 解析、报表生成、文档、样例和
  本地采集物料；
- `sglang_fl/profiling_hooks.py`：scheduler/TP worker 运行时需要 import、随 wheel 安装的
  最小 correlation hook；
- `tests/unit_tests/tools/operator_profiling/`：离线工具测试；
- `tests/unit_tests/platform/test_profiling_hooks.py`：plugin 运行时 hook 测试。

## 1. 当前统计范围

采集固定使用 `platform_profile` 和 eager 执行：

- 关闭 FlagGems 对 ATen 算子的替换；
- 不选择当前 FlagOS fused 实现；
- 保留 sglang-plugin-FL vendor dispatch；
- vendor 实现不可用时保留 SGLang 对当前平台选择的原生实现；
- 显式关闭全局 Torch Compile，框架内部局部 compile kernel 仍记录为 `torch_fused`；
- 关闭 CUDA Graph，避免只观察到 graph replay；
- warmup 在 profiler 启动前完成，不进入正式结果；
- 通信、GPU memcpy 和 GPU memset 不进入算子清单。

计算 kernel 按实际来源分成四类：

| `source_category` | 含义 |
| --- | --- |
| `torch_aten` | PyTorch ATen API 启动的 kernel |
| `torch_fused` | 框架内部局部 TorchInductor/compile 生成的 kernel |
| `third_party` | SGLang、sgl-kernel、FlashInfer 等上游组件的实现 |
| `vendor` | 实际进入 sglang-plugin-FL vendor dispatch 的实现 |

这里的来源表示采集时实际走到的执行路径。当前 `flagos/impl` 下调用
`flag_gems.modules` 或 `flag_gems.fused` 的实现仍可用于正常推理，但本工具不会把它们
声明为 FlagGems-sglang 覆盖。

## 2. 正式结果

每个 workload 的 `results/` 固定生成三个正式文件：

```text
results/
├── operator_list.csv
├── kernel_details_report.csv
└── summary.json
```

- `operator_list.csv`：去重后的 API 与物理 kernel 关系；
- `kernel_details_report.csv`：每种输入 shape/dtype 的次数、设备时间和来源；
- `summary.json`：环境、workload、四类来源汇总、逐 rank 汇总及校验结果。

正式 CSV 是 all-ranks 结果：名称和 shape 取 TP ranks 的并集，kernel 执行次数和设备
duration 跨 ranks 求和。`audit/operator_report/` 保存生成正式结果所依据的归一化中间表，
`traces/` 保存 torch profiler 原始 trace；两者不属于对外三文件。

仓库同时提供一份由真实 Qwen3.6-27B TP4 推理生成的小型
[`examples/qwen3_6_27b_tp4_smoke/`](examples/qwen3_6_27b_tp4_smoke/) 样例。它包含完整三文件，
可用于查看格式和直接运行独立校验；正式 1K/4K/16K 物料不提交源码 PR。

## 3. 环境准备与自动预检

下面的命令不会重新安装 SGLang、PyTorch、FlagTree 或 FlagGems，只把当前 checkout 以
editable 方式注册，并安装两个 SGLang plugin entry point：

```bash
cd /path/to/sglang-plugin-FL
python -m pip install -e . --no-deps
```

随后运行与正式采集相同的自动预检：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python tools/operator_profiling/check_environment.py \
  --model-path /models/Qwen3.6-27B \
  --tp-size 4 \
  --require-validated-versions
```

成功标志为 `OPERATOR_PROFILING_PREFLIGHT_PASS`。预检会检查：

- 模型目录存在；
- 当前 Python 能看到不少于 TP size 的 GPU；
- `sglang_fl` 的 platform/plugin entry point 都已注册；
- 实际 import 的 `sglang_fl` 来自当前 checkout；
- `sglang`、`torch`、`transformers` 和 `flagtree` distribution 均已安装；
- `triton` Python namespace 的实际 distribution provider 是 FlagTree；
- 当前 SGLang 源码能够使用已安装依赖导入 `Engine`，且提供 profiler 启停接口；
- 软件版本与本文验证基线一致。

`run_operator_profile.py` 在创建 Engine 和输出目录之前会自动执行同一套关键检查。预检
失败时不会启动模型，也不会生成看似成功的不完整物料。需要在其他版本组合上试运行时可
省略 `--require-validated-versions`；关键能力检查仍然执行，版本差异会明确打印为 warning。

## 4. 最小验证

下面的命令在四张卡上执行一个很小的真实模型 workload：

```bash
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

成功时最后输出：

```text
EAGER_OPERATOR_PROFILE_PASS output=<run-directory>
```

`--attention-backend triton` 是 SGLang v0.5.11 的 backend 注册名；当前验证环境由
FlagTree 0.6.2a1 承载这条编译路径。

## 5. 正式采集命令

当前正式输入为 1K、4K 和 16K，输出 1K，并发 64，TP4。256K 运行时间较长，暂不作为
本轮必跑项；工具仍支持在模型上下文允许的范围内传入 256K。

Qwen3.6-27B：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -u tools/operator_profiling/run_operator_profile.py \
  --model-path /models/Qwen3.6-27B \
  --tp-size 4 \
  --input-tokens 1024,4096,16384 \
  --output-tokens 1024 \
  --concurrency 64 \
  --mem-fraction-static 0.80 \
  --chunked-prefill-size 8192 \
  --attention-backend triton \
  --require-validated-versions \
  --output-dir tools/operator_profiling/runs/qwen3_6_27b
```

Qwen3.6-35B-A3B：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -u tools/operator_profiling/run_operator_profile.py \
  --model-path /models/Qwen3.6-35B-A3B \
  --tp-size 4 \
  --input-tokens 1024,4096,16384 \
  --output-tokens 1024 \
  --concurrency 64 \
  --mem-fraction-static 0.80 \
  --chunked-prefill-size 8192 \
  --attention-backend triton \
  --require-validated-versions \
  --output-dir tools/operator_profiling/runs/qwen3_6_35b_a3b
```

两组命令使用不同 GPU，可并行执行。每个输出根目录下会再创建 UTC 时间戳目录。

## 6. 已验证模型与产出

当前版本已在以下环境完成真实模型采集：

| 项目 | 版本或配置 |
| --- | --- |
| GPU | NVIDIA H20-3e，4 卡 |
| SGLang | 0.5.11 |
| PyTorch | 2.11.0+cu130 |
| FlagTree | 0.6.2a1 |
| 执行模式 | eager，TP4，全局 Torch Compile 关闭，CUDA Graph 关闭 |

正式验证矩阵如下。每个 workload 均生成三份正式文件，覆盖 TP ranks `[0,1,2,3]`，并通过
源报告守恒检查和三文件独立校验。

| 模型 | 输入 / 输出 | 并发 | 计算 kernel 执行次数（all ranks） | 结果 |
| --- | ---: | ---: | ---: | --- |
| Qwen3.6-27B | 1K / 1K | 64 | 4,947,436 | PASS |
| Qwen3.6-27B | 4K / 1K | 64 | 5,112,460 | PASS |
| Qwen3.6-27B | 16K / 1K | 64 | 5,769,740 | PASS |
| Qwen3.6-35B-A3B | 1K / 1K | 64 | 4,291,532 | PASS |
| Qwen3.6-35B-A3B | 4K / 1K | 64 | 4,431,116 | PASS |
| Qwen3.6-35B-A3B | 16K / 1K | 64 | 4,986,636 | PASS |

正式物料批次标识为 `platform_profile_h20_20260904`，当前收口代码在 PR #69 基线上的 TP4
smoke 批次标识为 `platform_profile_h20_20260908_pr69_smoke`。本地物料统一保存在
`tools/operator_profiling/materials/`；该目录由 `.gitignore` 排除，不纳入源码 PR。对外分发时应保留
目录结构以及 `summary.json` 中记录的 CSV 行数和 SHA-256。256K workload 尚未执行，不属于
本轮已验证范围。

## 7. 复用已有 profiler 审计结果

如果已经存在由 `trace_report` 生成的 `operator_list.csv`、
`kernel_shape_dtype.csv` 和 `profile_summary.json`，不需要重新执行模型，可直接整理为当前
三文件格式：

```bash
python tools/operator_profiling/format_operator_inventory.py \
  --source-dir /path/to/audited/operator_report \
  --output-dir /path/to/public/results
```

转换器首先检查原报告的全部 validation，然后再次检查 API/kernel 关系、shape JSON、
执行次数、设备时间和四类来源汇总是否守恒。任一检查失败都不会输出成功标志。

## 8. 结果验收

把下面的 `<results-directory>` 替换为某个 workload 的 `results/`：

```bash
python tools/operator_profiling/validate_operator_inventory.py <results-directory>
```

成功时输出 `OPERATOR_INVENTORY_VALIDATION_PASS`。该命令会重新读取落盘后的三个文件并
执行独立检查，包括：文件集合和表头、`platform_profile` 口径、API/kernel 来源关系、
shape/dtype JSON、非负数值、行唯一性、各行时间占比、总次数和总时间、四类
来源的次数和时间、TP ranks、warmup/profile 输出摘要、CSV 行数及 SHA-256。任一项不一致
都会以非零状态退出。

## 9. 主要参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model-path` | `/models/Qwen3.6-27B` | 本地模型目录 |
| `--tp-size` | `4` | TP rank 数 |
| `--input-tokens` | `1024,4096,16384` | 逗号分隔的请求输入长度；可显式传入 262144 |
| `--output-tokens` | `1024` | 每个请求生成的 token 数 |
| `--concurrency` | `64` | 同批请求数 |
| `--length-overflow-policy` | `truncate_input` | 超过模型上下文时截断输入；可改为 `error` |
| `--mem-fraction-static` | `0.80` | SGLang 静态显存比例 |
| `--chunked-prefill-size` | `8192` | chunked prefill 大小 |
| `--attention-backend` | `triton` | SGLang attention backend 注册名 |
| `--random-seed` | `0` | 随机种子；temperature 固定为 0 |
| `--require-validated-versions` | 关闭 | 要求 SGLang 0.5.11、PyTorch 2.11.0+cu130、FlagTree 0.6.2a1 |
| `--output-dir` | `tools/operator_profiling/runs` | 结果根目录 |

## 10. 后续算子库迁移

FlagGems-sglang 独立算子库的迁移仍属于后续工作。本次覆盖率工具统计当前插件实际使用的
`flag_gems.modules`、`flag_gems.fused` 和 ATen 替换实现，并支持有证据的显式融合映射。
未来迁移时仍需要完成以下事项：

- 将 `flagos/impl` 中当前使用的 `flag_gems.modules`、`flag_gems.fused` 实现逐项迁移到
  `flaggems_sglang`；
- 根据实际接入 API 建立 FlagGems-sglang 算子与原生 `vendor`、`torch_fused`、
  `third_party` 算子的明确映射；
- 记录实际选择的 provider callable、成功返回、fallback 和 blacklist；
- 校验相同 workload 下实际命中的 shape/dtype 变体；
- 为新增实现补充去重后的功能映射及运行证据。FlagOS 覆盖率按算子种类计算，调用次数与设备时间仅作为独立诊断指标。

原有 platform 物料只描述开发清单；覆盖率必须使用新工作流的 native/plugin 结果。

运行目录、trace 和生成物料不提交源码 PR；本地统一保存到仓库工作区内已忽略的
`tools/operator_profiling/runs/` 或 `tools/operator_profiling/materials/`，不使用 `/tmp` 作为交付位置。
