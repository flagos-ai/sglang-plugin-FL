# Qwen3.6-27B TP4 算子清单样例

这是一份由真实 SGLang 推理生成的小型算子清单，用于展示正式物料格式，并为报表解析方和
Reviewer 提供可直接执行的校验输入。样例不用于模型性能对比。

## 采集配置

| 项目 | 配置 |
| --- | --- |
| 模型 | Qwen3.6-27B |
| GPU | NVIDIA H20-3e × 4 |
| 并行方式 | TP4 |
| 输入 / 输出 | 16 / 4 tokens |
| 并发 | 2 |
| 执行模式 | `platform_profile`、eager、CUDA Graph 关闭 |
| SGLang | 0.5.11 |
| sglang-plugin-FL 基线 | 包含 PR #69 strict 语义修复的 main |
| PyTorch | 2.11.0+cu130 |
| FlagTree | 0.6.2a1 |

`summary.json` 中的主机名和输出目录已经替换为中性示例值；CSV 内容、CSV 行数、CSV
SHA-256 以及其余采集数据保持原始结果。样例保留实际设备时间，用于展示字段和校验关系，
不代表稳定的性能基线。

## 文件

```text
results/
├── operator_list.csv
├── kernel_details_report.csv
└── summary.json
```

- `operator_list.csv`：去重后的上层算子 API 与物理 kernel 关系；
- `kernel_details_report.csv`：按 API、kernel、shape 和 dtype 聚合的执行次数与设备时间；
- `summary.json`：采集配置、来源汇总、TP rank 汇总、CSV 校验信息。

完整字段定义和统计口径见 [`../../MATERIAL_GUIDE.md`](../../MATERIAL_GUIDE.md)。

## 校验

在仓库根目录执行：

```bash
python tools/operator_profiling/validate_operator_inventory.py \
  tools/operator_profiling/examples/qwen3_6_27b_tp4_smoke/results
```

成功时输出：

```text
OPERATOR_INVENTORY_VALIDATION_PASS \
  output=tools/operator_profiling/examples/qwen3_6_27b_tp4_smoke/results \
  checks=22
```

校验通过表示三文件 schema、行数、SHA-256 和各项汇总关系一致，不表示模型精度或性能已由
该样例完成验收。
