# 海光 BW1000：SGLang 0.5.18 empty 环境运行说明

[容器配方](empty-0.5.18.containerfile)保留原镜像的 DTK Torch、FlagTree、
FlagGems 和 native FlagCX。运行前需要设置以下环境变量；在宿主机直跑和进入
容器后执行 Python 时都应检查这些设置。

## 安装前应用 SGLang 源码补丁

必须对 **SGLang v0.5.18 源码**应用
[`sglang-v0.5.18-hcu.patch`](sglang-v0.5.18-hcu.patch)，然后再安装 SGLang。
仅安装或更新插件无法修复这一上游导入问题：量化注册表在插件补丁执行前就会
根据 HIP 标记导入 AMD Quark/AITER 内核；HCU 同样使用 HIP，但不提供这些内核。

容器配方负责这一安装顺序。手动从源码部署时，在安装 SGLang 前执行：

```bash
export PLUGIN_ROOT=/opt/sglang-plugin-fl
export SGLANG_SOURCE_ROOT=/opt/sglang-0.5.18
cd "$SGLANG_SOURCE_ROOT"
git apply --check "$PLUGIN_ROOT/docker/hygon/sglang-v0.5.18-hcu.patch"
git apply "$PLUGIN_ROOT/docker/hygon/sglang-v0.5.18-hcu.patch"
```

之后按容器配方安装该 SGLang 源码与插件。补丁仅让海光跳过三个不适用的 AMD
Quark 量化入口；HCU 的 HIP 判别保留，AMD 原有分支不变。若显式选择这些
Quark MXFP4/Int4FP8 路径，会得到说明 HCU 不支持的错误。BF16 模型导入不再
依赖这些 AMD 内核。不要通过安装 AMD AITER 覆盖基底设备栈。

源码 gate 回归可以在不加载 GPU 库的情况下执行；这不替代完整 Engine 验收：

```bash
cd "$PLUGIN_ROOT"
SGLANG_HCU_UPSTREAM_SOURCE="$SGLANG_SOURCE_ROOT" \
  python -m pytest -q tests/unit_tests/hygon/test_quark_source_guard.py
```

## Torch 自动加载与设备选择

基底镜像还安装了名为 `flagcx` 的 pip 包，其可选 Torch device extension 是
CUDA 构建。Torch 自动加载该扩展时会尝试加载 `libcudart.so.12`，导致海光环境
中 `import torch` 失败。必须在启动 Python 前设置：

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

插件显式加载自己的 native FlagCX wrapper；上述设置用于关闭 Torch 的可选设备
扩展自动加载，不会关闭插件的 FlagCX 通信路径。容器配方与
[`hygon` 测试配置](../../tests/platforms/hygon.yaml)均提供此默认值。
测试配置保留 shell 中已设置的值，因此仍应确认外部环境没有将它覆盖为 `1`。

海光设备筛选统一使用 `HIP_VISIBLE_DEVICES`。清除其它筛选变量，避免同一设备
列表被重复过滤。以下示例使用空闲的 2、3 号设备；按实际分配替换编号。
容器内编号取决于容器挂载的设备及其可见性映射。

```bash
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES=2,3
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export FLAGCX_PATH=/sgl-workspace/FlagCX
export SGLANG_FL_DIST_BACKEND=flagcx
```

## 环境检查与模型验收

在仓库根目录执行。`--baseline` 对应镜像构建时从原始设备栈保存的版本文件及
相邻 `.modules.json`；版本与模块 SHA256 检查不代表模型推理已经通过。

```bash
python .github/scripts/hygon/verify_environment.py \
  --plugin-root "$PWD" \
  --baseline /opt/hygon-vendor-baseline.json \
  --require-gpu --min-gpus 2

# TP2/PP1，eager decode；依次运行 inference、serving、concurrent。
python tests/run.py --platform hygon --device bw1000 --scope e2e

# TP2/PP1，标准 full decode graph；prefill 保持 eager。
python tests/run.py --platform hygon --device bw1000_graph --scope e2e
```

`--require-gpu` 实际比较海光 BF16 Torch、FlagOS `silu_and_mul` 和 reference
GemmaRMSNorm 的数值结果。empty 策略不选用 native `rms_norm`，验证器不会把
native RMSNorm ABI 或执行结果声称为已验证。完整 Engine、图重放及 TP/PP
通信应由相应模型验收确认。

保留上面的设备筛选和 Torch 环境设置后，也可以运行仓库示例：

```bash
MODEL_PATH=/models/Qwen3.6-27B TP_SIZE=2 \
  python examples/qwen3_6_27b_offline_inference.py

MODEL_PATH=/models/Qwen3.6-35B-A3B TP_SIZE=2 \
  python examples/qwen3_6_35b_a3b_offline_inference.py
```

这些验收配置不会自动选择 GPU。运行前确认 `HIP_VISIBLE_DEVICES` 对应已分配的
空闲设备；直接运行示例同样需要在 `import torch` 之前设置自动加载开关。
