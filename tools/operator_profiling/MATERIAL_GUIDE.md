# 算子清单物料说明

本文说明 `run_operator_profile.py` 生成的三个正式文件、字段含义、聚合方式和推荐阅读
顺序。当前物料描述的是采集时实际执行的原生平台算子，不包含 FlagGems-sglang 覆盖
结论。

## 1. 目录结构

一次运行可以包含多个输入长度。每个 workload 独立生成一组结果：

```text
<run-id>/
├── manifest.json
├── run.json
└── input_<requested>_actual_<actual>_output_<output>_concurrency_<n>/
    ├── COMPLETED
    ├── workload.json
    ├── results/
    │   ├── operator_list.csv
    │   ├── kernel_details_report.csv
    │   └── summary.json
    ├── audit/operator_report/
    └── traces/
```

需要交给算子团队的是 `results/` 中的三个文件。`audit/` 和 `traces/` 用于问题追溯，
体积可能较大，不提交源码仓库。

CSV 是纯文本格式，不支持字体、黑体、颜色等样式。第一行就是表头；是否显示为黑体由
Excel、LibreOffice 或网页查看器决定，不属于 CSV 文件内容。

## 2. 采集对象与聚合规则

正式 CSV 只保留计算 kernel。以下活动不进入清单：

- NCCL/FlagCX 等通信 kernel；
- GPU memcpy；
- GPU memset。

同一个 workload 的 TP ranks 按以下规则合并：

- API 名、kernel 名和 shape/dtype 变体取并集；
- 相同变体的 `kernel_event_count` 跨 ranks 求和；
- 相同变体的 `kernel_time_us` 跨 ranks 求和；
- 不对跨 stream、跨 rank 的设备 duration 做时间轴去重。

因此 `kernel_time_us` 是设备 kernel duration 的累计值，不是端到端延迟。多个 GPU、多个
stream 可以并行执行，累计设备时间可以大于请求墙钟时间。

## 3. API、kernel 和 shape 的关系

一行算子数据可能同时包含三层信息：

```text
operator_name  框架/运行库可观测到的 API 或算子入口
kernel_name    设备实际执行的稳定物理 kernel 名
input_shapes   profiler 关联到该次调用的输入 shape
```

例如：

```text
operator_name = aten::mm
kernel_name   = nvjet_sm90_...
input_shapes  = [[8192,5120],[5120,8704]]
```

表示 `aten::mm` 的这一组输入最终启动了对应的设备 kernel。

对于 fused、third-party 或 vendor 实现，API 与 kernel 未必一一对应：

- 一个 API 可以启动多个 kernel；
- 一个 kernel 可以有多个 shape/dtype 变体；
- 同一个稳定 kernel 名可能被多个 API 调用。

工具保留实际观察到的关系，不根据名称相似度推断不存在的映射。

## 4. `operator_list.csv`

该文件是去重后的 API 与物理 kernel 关系表，适合快速查看需要开发或优化的算子范围。

字段顺序为：

```text
operator_id,operator_name,operator_kind,kernel_name,source_category,source_library
```

前四列保持既有参考物料的顺序；仅在末尾追加本工具需要的来源类别和实现库。

### 4.1 字段解释

| 字段 | 含义 |
| --- | --- |
| `operator_id` | 本次 workload 内的稳定编号。ATen 按 API 名归并；其他实现通常按稳定 kernel 名归并。编号只在当前文件内有效 |
| `operator_name` | profiler 关联到的 API 或运行时算子名，例如 `aten::mm`、`sglang::store_cache`、某个 Triton/JIT callable |
| `operator_kind` | API/实现形态，例如 `aten`、`custom`、`torch_compile`、`unattributed` |
| `kernel_name` | 设备实际执行的稳定 kernel 名。动态地址、clone 后缀等不稳定部分已经归一化 |
| `source_category` | 实现来源类别：`torch_aten`、`torch_fused`、`third_party` 或 `vendor` |
| `source_library` | 更具体的实现库，例如 `pytorch`、`flashinfer`、`sgl_kernel`、`sglang`、`cuda` |

### 4.2 `operator_kind`

常见值：

| 值 | 含义 |
| --- | --- |
| `aten` | 可关联到明确 ATen API |
| `custom` | 自定义算子、第三方算子或 JIT 算子入口 |
| `torch_compile` | 可由 profiler 元数据确认来自局部 TorchInductor compile |
| `fused_communication_compute` | 同时包含通信与计算语义、按规则保留的融合计算 kernel |
| `unattributed` | 找到了计算 kernel，但 profiler 没有提供可靠的上层 API 名 |
| `unattributed_nvjet` | 无可靠 API 名且属于 nvjet kernel 的特殊标识 |

`unattributed` 行不会被删除，因为它仍然是算子团队可能需要处理的真实计算 kernel。

### 4.3 `source_category`

| 值 | 判断依据 |
| --- | --- |
| `torch_aten` | 上层 operator 是明确的 `aten::` API |
| `torch_fused` | profiler CPU 事件包含 TorchInductor kernel hash/backend/file 元数据 |
| `third_party` | 运行时名称、命名空间或实现元数据显示来自 SGLang、sgl-kernel、FlashInfer 等上游库 |
| `vendor` | kernel 位于实际执行的 plugin `VENDOR OpImpl` 调用范围内；不是根据 CUDA/CUTLASS 字符串猜测 |

在 NVIDIA 平台上走 `vendor.cuda` 后，即使 adapter 内部委托了 SGLang/sgl-kernel，按已经
确定的 dispatch 路径口径仍归为 `vendor`。具体委托信息保存在审计报告，不在对外简表中
重复展开。

## 5. `kernel_details_report.csv`

该文件是算子开发的主要输入。相同 API/kernel 的不同 shape 或 dtype 分别占一行。

前九列与既有参考物料保持一致：

```text
operator_name,kernel_name,variant_index,mapping_status,input_shapes,
input_dtypes,candidate_operators,kernel_event_count,kernel_time_us
```

后面追加实现来源和优先级信息：

```text
operator_kind,source_category,source_library,execution_origin,
kernel_time_percent,kernel_time_percent_of_category
```

### 5.1 字段解释

| 字段 | 含义 |
| --- | --- |
| `operator_name` | 与该 kernel launch 关联的 API 或运行时算子名 |
| `kernel_name` | 设备实际执行的稳定 kernel 名 |
| `variant_index` | 同一稳定 kernel 的 shape/dtype 变体编号；按累计设备时间从高到低排列 |
| `mapping_status` | API、kernel 和 shape 关联证据的状态，见下节 |
| `input_shapes` | JSON 数组。每一项对应一个输入参数；Tensor 保存维度，标量通常保存空数组 |
| `input_dtypes` | JSON 数组，与 `input_shapes` 同位置对应；Tensor 保存 dtype，标量保存类型说明 |
| `candidate_operators` | 当一个 kernel launch 可能关联多个上层 operator 时保存候选 API 的 JSON 数组；没有歧义时为 `null` |
| `kernel_event_count` | 该变体在正式 profiler 窗口内被观察到的 kernel 执行次数，跨 TP ranks 求和 |
| `kernel_time_us` | 该变体所有 kernel duration 的总和，单位微秒，跨 TP ranks 求和 |
| `operator_kind` | 与 `operator_list.csv` 相同的实现形态 |
| `source_category` | 四类来源之一 |
| `source_library` | 更具体的实现库；极少数多来源关系以 JSON 数组表示 |
| `execution_origin` | `eager` 或 `torch_compile`；这里的 `torch_compile` 指框架内部实际产生的局部 compile kernel |
| `kernel_time_percent` | `kernel_time_us / 全部纳入清单的计算 kernel 时间 × 100`，数值范围 0–100，不带 `%` 字符 |
| `kernel_time_percent_of_category` | `kernel_time_us / 当前 source_category 的计算 kernel 时间 × 100`，数值范围 0–100 |

### 5.2 `mapping_status`

常见值：

| 值 | 含义 |
| --- | --- |
| `operator_shape_matched` | 通过 profiler correlation 将 kernel 与上层 operator、shape/dtype 关联 |
| `native_marker_operator_shape_matched` | 通过本工具增加的 `record_function` marker 关联到 SGLang/JIT/plugin 调用 |
| `operator_matched_shape_missing` | 找到上层 operator，但 profiler 没有提供完整 shape |
| `ambiguous_operator_match` | 同一个 launch 存在多个无法唯一消解的候选 operator；候选保存在 `candidate_operators` |
| `no_cpu_op_match` | 没有找到可靠的上层 CPU operator，`operator_name` 可能为 `null` |

所有映射状态都会保留。查看算子清单时应优先使用明确 matched 的行，再检查 ambiguous 或
missing 行，不能把 `null` 理解为“没有执行”。

### 5.3 shape 的限制

`input_shapes` 来自 torch profiler 或明确的运行时 marker，表示 launch 关联到的输入，
不是对算子数学语义的人工定义。某些 C++/第三方库只向 profiler 暴露部分参数，因此可能
出现：

- shape 为空；
- 仅有部分 Tensor 参数；
- 标量只记录类型，不记录具体值；
- 候选 operator 多于一个。

这些情况通过 `mapping_status` 明确呈现，不会由工具补猜。

## 6. `summary.json`

该文件用于确认采集条件和 CSV 守恒关系。顶层字段如下。

| 字段 | 含义 |
| --- | --- |
| `schema_version` | 当前正式物料 schema 版本 |
| `report_type` | 固定为 `operator_inventory` |
| `scope` | 执行模式、`platform_profile` 口径、排除项和聚合规则 |
| `metadata` | 机器、软件版本、SGLang 参数和实际 workload |
| `ranks` | 纳入合并的 TP rank 编号，TP4 应为 `[0,1,2,3]` |
| `compute_kernel_event_count` | 正式 CSV 中计算 kernel 次数总和 |
| `compute_kernel_time_us` | 正式 CSV 中计算 kernel duration 总和 |
| `unique_stable_kernel_names` | 去重后的稳定 kernel 名数量 |
| `source_category_*` | 按四类来源汇总的去重数、次数、时间和占比 |
| `source_library_*` | 按实现库汇总的次数和时间 |
| `execution_origin_*` | 按 eager/局部 compile 汇总的次数和时间 |
| `mapping_*` | 按 `mapping_status` 汇总的次数和时间 |
| `excluded_*` | 被明确排除的通信、memcpy、memset 等次数和时间 |
| `dispatch` | 实际 plugin dispatch 调用和物理 kernel 审计摘要 |
| `torch_compile` | 局部 TorchInductor kernel 的识别方法、数量和时间 |
| `per_rank` | 每个 rank 的事件数和时间，便于发现负载异常 |
| `source_report_validation` | 原始 trace 归一化报告的守恒检查 |
| `validation` | 三个正式文件的结构和守恒检查 |
| `files` | 两个 CSV 的行数和 SHA-256 |

### 6.1 `metadata.workload`

重点字段：

| 字段 | 含义 |
| --- | --- |
| `requested_input_tokens` | 命令要求的输入长度 |
| `actual_input_tokens` | 考虑模型上下文限制后实际执行的输入长度 |
| `requested_output_tokens` | 命令要求的输出长度 |
| `actual_output_tokens` | 实际要求模型生成的输出长度 |
| `concurrency` | 同一批请求数量 |
| `warmup_included` | 固定为 `false` |
| `warmup_seconds` | warmup 请求墙钟时间，不进入 kernel 统计 |
| `profiled_wall_time_seconds` | 正式批次墙钟时间 |
| `profiled_output_sha256` | 正式批次生成文本摘要，用于确认确定性运行 |
| `profiled_output_token_prefixes` | 新采集版本的可选字段：每个请求前若干输出 token，便于快速人工检查；由旧审计结果转换的物料可能没有该字段 |

`requested_input_tokens` 与 `actual_input_tokens` 不同表示输入按配置被截断。查看长上下文
物料时必须以 `actual_input_tokens` 为准。

### 6.2 `validation`

正式结果只有在全部字段为 `true` 时才会输出成功标志：

| 检查 | 含义 |
| --- | --- |
| `source_report_passed` | 原始 profiler 报告全部校验通过 |
| `platform_profile_configuration_is_valid` | 已确认 FlagGems/FlagOS fused 关闭、vendor dispatch 保留、全局 compile/CUDA Graph 关闭 |
| `operator_list_is_not_empty` | API/kernel 关系表非空 |
| `kernel_details_are_not_empty` | shape/dtype 明细非空 |
| `all_details_have_operator_provenance` | 每条明细都能在 operator 表找到来源关系 |
| `all_source_categories_are_valid` | 所有行只属于约定的四类来源 |
| `all_shape_fields_are_json_or_null` | shape、dtype 和候选 API 字段均可按 JSON 解析或为 `null` |
| `kernel_event_count_is_conserved` | 明细次数之和等于原始报告总次数 |
| `kernel_time_is_conserved` | 明细设备时间之和等于原始报告总时间 |
| `source_category_event_count_is_conserved` | 每个来源类别的次数保持一致 |
| `source_category_time_is_conserved` | 每个来源类别的设备时间保持一致 |

这些 `PASS/true` 只表示采集和报表聚合一致，不表示某个算子数值精度已经验收，也不表示
FlagGems 或 FlagGems-sglang 已经覆盖该算子。

落盘后可再次执行独立验收：

```bash
python tools/operator_profiling/validate_operator_inventory.py <results-directory>
```

该命令还会核对固定三文件、正式表头、`profile_mode=platform_profile`、行唯一性、
非负数值、逐行时间占比、TP ranks、warmup/profile 输出摘要、CSV 行数与
SHA-256。成功标志为 `OPERATOR_INVENTORY_VALIDATION_PASS`。

## 7. 推荐阅读顺序

1. 查看 `summary.json.metadata`，确认模型、SGLang/torch/FlagTree 版本和 TP 配置；
2. 查看 `summary.json.metadata.workload`，确认实际输入、输出和并发；
3. 确认 `source_report_validation` 与 `validation` 全部为 `true`；
4. 从 `operator_list.csv` 查看 API、kernel 和来源类别；
5. 从 `kernel_details_report.csv` 按 `kernel_time_us` 或 `kernel_event_count` 排序；
6. 对重点算子查看全部 shape/dtype 变体和 `mapping_status`；
7. 对 `ambiguous`、`missing`、`null` 行回看 audit 或原始 trace。

## 8. 当前不应从物料中得出的结论

当前物料不能用于直接宣称：

- FlagGems-sglang 已经覆盖某个 fused 算子；
- 名字相似的两个算子具有相同功能；
- 一个 provider 算子完整替换了某个 vendor/third-party/torch-fused 实现；
- `kernel_time_us` 等于请求端到端延迟；
- 跨 stream kernel 时间占比可以直接解释为墙钟时间占比。

FlagGems-sglang 接入后的覆盖映射、运行时命中和覆盖率计算列在 README 的 TODO 中。
