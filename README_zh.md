# sglang-plugin-FL

sglang-plugin-FL 是 [SGLang](https://github.com/sgl-project/sglang) 的 OOT（Out-of-Tree）插件，基于 FlagOS 统一多芯后端构建——包括统一算子库 [FlagGems](https://github.com/flagos-ai/FlagGems) 和统一通信库 [FlagCX](https://github.com/flagos-ai/FlagCX)。它将 SGLang 的推理能力扩展到多种硬件平台。无需修改 SGLang 的原始接口或使用方式，同一条命令即可在不同芯片上运行模型推理。

## 概述

SGLang 的推理引擎依赖 NVIDIA 专有组件：flashinfer 用于 attention、sgl_kernel 用于融合 CUDA 算子、NCCL 用于分布式通信。在其他硬件（华为昇腾、寒武纪 MLU、天数智芯等）上运行通常需要侵入式的源码修改。

本插件通过三层替换提供无侵入的适配层：

- **Layer 1 — ATen 算子**：通过 PyTorch dispatch 机制，将 PyTorch 底层算子（matmul、softmax、embedding 等）替换为 FlagGems Triton kernel
- **Layer 2 — SGLang 融合算子**：通过 HookRegistry AROUND hook 拦截 SGLang 自定义融合算子（SiluAndMul、RMSNorm、RotaryEmbedding），经标准化 dispatch 系统（与 vllm-plugin-FL 对齐）路由到 FlagGems、厂商原生或 PyTorch 参考实现
- **Layer 3 — 分布式通信**：将基于 NCCL 的集合通信替换为 CommunicatorFL（底层使用 FlagCX 或 torch.distributed），支持任意硬件的多卡推理

```
┌──────────────────────────────────────────────────────────────┐
│                       SGLang Runtime                         │
├──────────────────────────────────────────────────────────────┤
│  Layer 1: ATen Ops (flag_gems.enable → PyTorch dispatch)     │
│    torch.mm / torch.add / torch.softmax / ...                │
│      → FlagGems Triton kernels                               │
├──────────────────────────────────────────────────────────────┤
│  Layer 2: SGLang Fused Ops (AROUND hook on dispatch_forward) │
│    SiluAndMul / RMSNorm / RotaryEmbedding                    │
│      → flagos (FlagGems Triton) | vendor (chip-native) | ref │
├──────────────────────────────────────────────────────────────┤
│  Layer 3: Communication (AROUND hooks on GroupCoordinator)   │
│    all_reduce / all_gather / reduce_scatter / send / recv    │
│      → CommunicatorFL (FlagCX / torch.distributed)           │
├──────────────────────────────────────────────────────────────┤
│  Triton JIT / Vendor Native → GPU / NPU Kernels              │
└──────────────────────────────────────────────────────────────┘
```

芯片厂商只需实现一个 backend class + `register_ops.py`，dispatch 系统的自动发现机制会处理其余工作。同一份厂商实现可在 sglang-plugin-FL 和 vllm-plugin-FL 之间零改动复用。

## 环境依赖

| 组件 | 版本 |
|------|------|
| SGLang | 0.5.11 |
| sglang-kernel | 0.4.2 |
| PyTorch | 2.11.0+cu130 |
| Triton | 3.6.0 |
| FlagGems | 4.2.1rc0 |
| flashinfer | 0.6.8.post1 |
| Python | 3.12 |
| CUDA | 13.0 |

## 已验证模型

理论上 sglang-plugin-FL 可以支持 SGLang 中所有模型，只要不涉及未支持的算子。

| 模型 | TP | 状态 |
|------|-----|------|
| Qwen2.5-0.5B-Instruct | tp=1 | 已验证 |
| Qwen2.5-14B-Instruct | tp=8 | 已验证 |
| Qwen3.6-27B | tp=1 | 已验证 |
| Qwen3.6-35B-A3B | tp=1 | 已验证 |

## 快速开始

### 安装

1. 安装 SGLang v0.5.11：

```bash
pip install "sglang[all]==0.5.11"
```

2. 安装 [FlagGems](https://github.com/flagos-ai/FlagGems)：

```bash
git clone https://github.com/flagos-ai/FlagGems
cd FlagGems && pip install .
```

3. 安装本插件：

```bash
git clone https://github.com/flagos-ai/sglang-plugin-FL
cd sglang-plugin-FL && pip install .
```

4. （可选）安装 [FlagCX](https://github.com/flagos-ai/FlagCX) 用于多芯分布式通信：

```bash
git clone https://github.com/flagos-ai/FlagCX.git
cd FlagCX && make USE_NVIDIA=1
export FLAGCX_PATH="$PWD"
```

### 下载模型

```bash
# 小模型用于快速测试（单卡）
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct

# 大模型用于多卡（tp=8）
huggingface-cli download Qwen/Qwen2.5-14B-Instruct
```

如果无法访问 HuggingFace，使用镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
```

模型默认缓存在 `~/.cache/huggingface/hub/`。也可以直接传本地路径给 `--model-path`。

### 运行推理

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 \
    --disable-piecewise-cuda-graph
```

多卡张量并行：

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-14B-Instruct \
    --tp 8 --port 30000 \
    --disable-piecewise-cuda-graph
```

服务就绪后（`The server is fired up and ready to roll`），发送请求：

```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "列出前5个素数。"}],
    "temperature": 0
  }' | python -m json.tool
```

### 使用原生 CUDA 算子

禁用插件，使用 SGLang 原始 CUDA 路径：

```bash
SGLANG_PLUGINS="__none__" python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph
```

仅禁用 ATen 层（保留融合算子 dispatch）：

```bash
USE_FLAGGEMS=0 python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph
```

## 高级配置

大多数场景下默认配置即可使用。需要自定义时，使用 YAML 配置文件。

### YAML 配置文件

插件内置了 [`config/sample.yaml`](sglang_fl/config/sample.yaml)，包含所有可用选项。复制并自定义：

```bash
# 复制示例配置
cp $(python -c "from sglang_fl.config import _CONFIG_DIR; print(_CONFIG_DIR / 'sample.yaml')") my_config.yaml

# 按需编辑后启动
SGLANG_FL_CONFIG=./my_config.yaml python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph
```

如果未设置 `SGLANG_FL_CONFIG`，插件使用默认配置（在 CUDA 上等同于 `prefer: flagos`）。只有需要自定义行为时才需要 YAML 文件。

#### 配置字段

```yaml
# 全局后端优先级：flagos | vendor | reference
prefer: flagos

# 按算子后端优先级（有序列表，第一个可用的生效）
op_backends:
  rms_norm: [vendor, flagos, reference]
  silu_and_mul: [flagos, vendor, reference]

# Layer 2 融合算子跳过列表（回退到 SGLang 原生 CUDA）
# 可选：SiluAndMul, RMSNorm, RotaryEmbedding
oot_blacklist:
  - RotaryEmbedding

# Layer 1 ATen 算子排除列表（不使用 FlagGems Triton 替换）
flagos_blacklist:
  - mul
  - sub
```

| 字段 | 说明 |
|------|------|
| `prefer` | 全局后端优先级：`flagos`、`vendor`、`reference` |
| `op_backends` | 按算子有序后端列表（第一个可用的生效，可列 1-3 个后端） |
| `oot_blacklist` | Layer 2 融合算子跳过列表（回退到 SGLang 原生 CUDA） |
| `flagos_blacklist` | Layer 1 ATen 算子排除列表（回退到 PyTorch 原生实现） |

#### 常用配置示例

每个示例展示 YAML 配置和预期 dispatch 结果。使用 [Dispatch 日志](#dispatch-日志) 验证。

**1. 跳过 RotaryEmbedding 的 OOT dispatch**（回退到 SGLang 原生 CUDA）：

```yaml
# my_config.yaml
prefer: flagos
oot_blacklist:
  - RotaryEmbedding
```

预期 dispatch 日志：只出现 SiluAndMul 和 RMSNorm，无 RotaryEmbedding。

**2. 强制 RMSNorm 使用 vendor 后端，其他使用 flagos：**

```yaml
# my_config.yaml
prefer: flagos
op_backends:
  rms_norm: [vendor, flagos, reference]
```

预期 dispatch 日志：`RMSNorm → vendor(vendor.nvidia)`，`SiluAndMul → flagos(flagos)`。

**3. 所有算子使用纯 PyTorch 参考实现**（精度调试场景）：

```yaml
# my_config.yaml
prefer: reference
```

预期 dispatch 日志：所有算子 → `reference(reference)`。

### 环境变量 — 完整参考

所有插件行为通过 `SGLANG_FL_*` 环境变量控制，优先级高于 YAML 配置。

**优先级链：**
```
SGLANG_FL_* 环境变量 > YAML 配置 (SGLANG_FL_CONFIG) > 平台自动检测 YAML > 代码默认值
```

#### Layer 2 — 融合算子 Dispatch

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SGLANG_FL_OOT_ENABLED` | `1` | 总开关：`0` 禁用 Layer 2（Layer 1 ATen 保持激活） |
| `SGLANG_FL_PREFER` | `flagos` | 全局后端优先级：`flagos`、`vendor`、`reference` |
| `SGLANG_FL_PER_OP` | — | 按算子后端优先级，如 `rms_norm=vendor\|flagos;silu_and_mul=reference` |
| `SGLANG_FL_OOT_BLACKLIST` | — | 跳过列表（逗号分隔的类名） |
| `SGLANG_FL_OOT_WHITELIST` | — | 白名单：仅列出的算子走 OOT dispatch（与 BLACKLIST 互斥） |
| `SGLANG_FL_STRICT` | `0` | `1` = 禁用回退（首选后端不可用时报错） |
| `SGLANG_FL_DENY_VENDORS` | — | 禁用指定厂商（逗号分隔，如 `cuda,ascend`） |
| `SGLANG_FL_ALLOW_VENDORS` | — | 仅允许列出的厂商（逗号分隔） |
| `SGLANG_FL_DISPATCH_LOG` | — | Dispatch 日志文件路径（记录哪些算子被拦截） |

#### Layer 1 — ATen 替换（FlagGems）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `USE_FLAGGEMS` | `1` | 总开关：`0` 禁用所有 ATen 替换 |
| `SGLANG_FL_FLAGOS_WHITELIST` | — | 仅列出的 ATen 算子使用 FlagGems（逗号分隔） |
| `SGLANG_FL_FLAGOS_BLACKLIST` | — | 列出的 ATen 算子不使用 FlagGems（逗号分隔） |
| `SGLANG_FLAGGEMS_RECORD` | `0` | `1` = 记录被替换的 ATen 算子 |
| `SGLANG_FLAGGEMS_LOG_PATH` | — | ATen 替换日志文件路径 |
| `SGLANG_FLAGGEMS_LOG_ONCE` | `1` | `1` = 每个算子只记录一次，`0` = 每次调用都记录 |

> `FLAGOS_WHITELIST` 和 `FLAGOS_BLACKLIST` 互斥。`FLAGOS_WHITELIST` 优先级高于 YAML 中的 `flagos_blacklist`。

#### Layer 3 — 分布式通信

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SGLANG_FL_DIST_BACKEND` | `nccl` | 通信后端：`nccl` / `hccl` / `flagcx` |
| `FLAGCX_PATH` | — | FlagCX 安装路径（设置后默认使用 `flagcx` 后端） |

#### 系统 / 调试

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SGLANG_FL_CONFIG` | — | YAML 配置文件路径（覆盖平台自动检测） |
| `SGLANG_FL_PLATFORM` | (自动) | 强制指定平台：`cuda`、`ascend`（覆盖自动检测） |
| `SGLANG_FL_LOG_LEVEL` | `INFO` | Dispatch 系统日志级别：`DEBUG`、`INFO`、`WARNING`、`ERROR` |
| `SGLANG_PLUGINS` | (全部) | SGLang 内置：过滤加载哪些插件（逗号分隔）。通常不需要——`pip install` 后插件自动发现 |

#### 示例

```bash
# 所有算子使用 reference 后端（纯 PyTorch，精度调试场景）
SGLANG_FL_PREFER=reference python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

# 按算子：RMSNorm 使用 vendor，其他使用 flagos
SGLANG_FL_PER_OP="rms_norm=vendor|flagos;silu_and_mul=flagos" \
    python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

# 跳过 RotaryEmbedding（回退到 SGLang 原生 CUDA）
SGLANG_FL_OOT_BLACKLIST=RotaryEmbedding python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

# 禁用 ATen 层，仅保留融合算子 dispatch
USE_FLAGGEMS=0 python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

# 使用 YAML 配置 + 环境变量覆盖
SGLANG_FL_CONFIG=./my_config.yaml SGLANG_FL_PREFER=reference \
    python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph
```

## 调试与诊断

### Dispatch 日志

查看每个融合算子解析到了哪个后端（服务启动时写入）：

```bash
rm -f /tmp/dispatch.log
SGLANG_FL_DISPATCH_LOG=/tmp/dispatch.log \
  python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

sort -u /tmp/dispatch.log
# [OOT-DISPATCH] SiluAndMul → flagos(flagos)
# [OOT-DISPATCH] RMSNorm → flagos(flagos)
# [OOT-DISPATCH] RotaryEmbedding → flagos(flagos)
```

### ATen 替换日志

记录哪些 PyTorch ATen 算子被 FlagGems 替换：

```bash
rm -f /tmp/gems_aten.txt
SGLANG_FLAGGEMS_RECORD=1 SGLANG_FLAGGEMS_LOG_PATH=/tmp/gems_aten.txt \
  python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

# 第一次推理请求后：
sort -u /tmp/gems_aten.txt
```

> 日志使用 `_AtenOnlyFilter` 仅记录 `flag_gems.ops.*` 命名空间的调用，排除 Layer 2 flagos 实现触发的内部 FlagGems 调用。

### 精度二分法

当出现数值差异时，逐层隔离定位问题：

```bash
# Step 1：禁用所有——确认原生 SGLang 正常
SGLANG_PLUGINS="__none__" python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

# Step 2：仅启用 Layer 2（融合算子），禁用 ATen 替换
USE_FLAGGEMS=0 python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

# Step 3：按算子隔离——仅 SiluAndMul 使用 flagos，RMSNorm 使用 reference
USE_FLAGGEMS=0 \
SGLANG_FL_PER_OP="silu_and_mul=flagos;rms_norm=reference" \
    python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

# Step 4：禁用 Layer 2，仅 ATen 替换激活
SGLANG_FL_OOT_ENABLED=0 python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

# Step 5：逐步启用 ATen 算子（通过白名单）
SGLANG_FL_OOT_ENABLED=0 SGLANG_FL_FLAGOS_WHITELIST=rms_norm,silu \
    python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph
```

> 如果在 Step N 出现输出差异但 Step N-1 正常，则问题层/算子已被隔离。

### 常见问题

| 现象 | 原因与解决 |
|------|-----------|
| `dispatch.log` 为空 | 插件未加载——检查 `pip show sglang_fl` |
| `gems_aten.txt` 为空 | 设置了 `USE_FLAGGEMS=0`，或 `SGLANG_FL_FLAGOS_WHITELIST` 排除了该算子 |
| 非 NVIDIA 上 `forward_cuda` 报错 | 某算子缺少 OOT 注册——注册该算子或加入白名单 |
| `ImportError: sgl_kernel` | 非 CUDA 平台正常现象——OOT dispatch 会绕过 `forward_cuda` |
| tp>1 启动时卡住 | 检查 GPU 数量、NCCL 环境变量、模型 TP 兼容性 |
| 引擎启动时 OOM | 减小 `--mem-fraction-static`（默认 0.5） |

## 厂商接入

芯片厂商通过在 `dispatch/backends/vendor/` 下添加 backend 目录来接入：

```bash
cp -r sglang_fl/dispatch/backends/vendor/template/ \
      sglang_fl/dispatch/backends/vendor/my_chip/
```

需要实现两个文件：

### 1. Backend 类（`my_chip.py`）

```python
from sglang_fl.dispatch.backends import Backend


class MyChipBackend(Backend):
    _available = None

    @property
    def name(self) -> str:
        return "my_chip"

    def is_available(self) -> bool:
        if MyChipBackend._available is None:
            try:
                import my_chip_sdk
                MyChipBackend._available = my_chip_sdk.device_count() > 0
            except ImportError:
                MyChipBackend._available = False
        return MyChipBackend._available

    def silu_and_mul(self, obj, x):
        from .impl.activation import silu_and_mul_my_chip
        return silu_and_mul_my_chip(obj, x)

    def rms_norm(self, obj, x, residual=None):
        from .impl.normalization import rms_norm_my_chip
        return rms_norm_my_chip(obj, x, residual)

    def rotary_embedding(self, obj, query, key, cos, sin, position_ids,
                         rotary_interleaved=False, inplace=True):
        from .impl.rotary import rotary_embedding_my_chip
        return rotary_embedding_my_chip(
            obj, query, key, cos, sin, position_ids, rotary_interleaved, inplace
        )
```

### 2. 注册文件（`register_ops.py`）

```python
import functools
from sglang_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority


def _bind_is_available(fn, is_available_fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    from .my_chip import MyChipBackend

    backend = MyChipBackend()
    is_avail = backend.is_available

    impls = [
        OpImpl(
            op_name="silu_and_mul",
            impl_id="vendor.my_chip",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.silu_and_mul, is_avail),
            vendor="my_chip",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="rms_norm",
            impl_id="vendor.my_chip",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rms_norm, is_avail),
            vendor="my_chip",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="rotary_embedding",
            impl_id="vendor.my_chip",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rotary_embedding, is_avail),
            vendor="my_chip",
            priority=BackendPriority.VENDOR,
        ),
    ]
    registry.register_many(impls)
```

### 3. 算子实现（`impl/`）

每个算子函数接收标准化参数（与 vllm-plugin-FL 一致）：

| 算子 | 签名 |
|------|------|
| `silu_and_mul` | `fn(obj, x: Tensor) -> Tensor` |
| `rms_norm` | `fn(obj, x: Tensor, residual: Optional[Tensor] = None) -> Tensor \| tuple[Tensor, Tensor]` |
| `rotary_embedding` | `fn(obj, query, key, cos, sin, position_ids, rotary_interleaved=False, inplace=True) -> tuple[Tensor, Tensor]` |

`obj` 参数提供对 layer 属性的访问（`obj.weight`、`obj.variance_epsilon` 等）。这些属性名在 SGLang 和 vLLM 之间完全一致，因此同一份 impl 可在两个框架上使用。

### 自动发现

插件启动时扫描 `dispatch/backends/vendor/*/register_ops.py`。如果 `is_available()` 返回 True，该厂商的算子会被注册。无需修改其他文件。

### 已有后端

| 厂商 | 目录 | 硬件检测 |
|------|------|---------|
| NVIDIA CUDA | `vendor/cuda/` | `sgl_kernel` 可导入 |
| 华为昇腾 | `vendor/ascend/` | `torch_npu` 可导入 |
| 模板 | `vendor/template/` | 始终 False（仅供参考） |

## 项目结构

```
sglang_fl/
├── pyproject.toml                    # 包配置 + entry_points 注册
└── sglang_fl/
    ├── __init__.py                   # 插件入口：FlagGems + dispatch 初始化 + communicator hooks
    ├── platform.py                   # PlatformFL（设备标识、内存、graph capture）
    ├── distributed/                  # 通信模块（与 vllm-plugin-FL 对齐）
    │   ├── __init__.py
    │   ├── communicator.py           # CommunicatorFL（FlagCX / torch.distributed 封装）
    │   └── device_communicators/
    │       └── flagcx.py             # FlagCX 专用 communicator
    ├── config/
    │   ├── __init__.py               # YAML 配置加载器，支持平台自动检测
    │   ├── sample.yaml               # 完整示例配置，包含所有选项说明
    │   ├── nvidia.yaml               # NVIDIA CUDA 平台默认配置
    │   └── ascend.yaml               # 昇腾平台默认配置
    └── dispatch/                     # 算子 dispatch 系统（与 vllm-plugin-FL 对齐）
        ├── __init__.py               # 公共 API：call_op()、resolve_op()
        ├── types.py                  # OpImpl、BackendImplKind、BackendPriority
        ├── registry.py               # 线程安全的 OpRegistry
        ├── policy.py                 # SelectionPolicy + 环境变量 / YAML 配置
        ├── manager.py                # OpManager：resolve、call、cache、fallback
        ├── builtin_ops.py            # 注册编排器
        ├── ops.py                    # FLBackendBase ABC（算子签名定义）
        ├── logger_manager.py         # 日志管理，支持 SGLANG_FL_LOG_LEVEL
        ├── bridge/                   # SGLang ↔ dispatch 参数转换层
        │   ├── __init__.py
        │   ├── silu_and_mul.py       # forward_cuda(self, x) → call_op("silu_and_mul", obj, x)
        │   ├── rms_norm.py           # 处理 post_residual_addition
        │   └── rotary_embedding.py   # 从 cos_sin_cache 提取 cos/sin，处理 offsets
        └── backends/
            ├── __init__.py           # Backend ABC
            ├── flaggems/             # DEFAULT 后端（FlagGems Triton kernels）
            │   ├── flaggems.py
            │   ├── register_ops.py
            │   └── impl/             # activation.py, normalization.py, rotary.py
            ├── reference/            # REFERENCE 后端（PyTorch 原生，始终可用）
            │   ├── reference.py
            │   ├── register_ops.py
            │   └── impl/             # activation.py, normalization.py, rotary.py
            └── vendor/               # VENDOR 后端（自动发现）
                ├── ascend/           # 华为昇腾 NPU (torch_npu)
                ├── cuda/             # NVIDIA CUDA (sgl_kernel)
                └── template/         # 新厂商模板
```

## 工作原理

### 插件加载

插件在 `pyproject.toml` 中注册两个 entry_points：

```toml
[project.entry-points."sglang.srt.plugins"]
sglang_fl = "sglang_fl:load_plugin"

[project.entry-points."sglang.srt.platforms"]
sglang_fl = "sglang_fl:activate_platform"
```

SGLang 启动时通过 setuptools entry_points 自动发现并加载插件。

### Dispatch Hook

核心机制使用 `MultiPlatformOp.dispatch_forward()` 上的 AROUND hook 结合标准化 dispatch 系统：

```
dispatch_forward() 被调用（如 RMSNorm）
  → AROUND hook 拦截
    → 检查 OOT_WHITELIST/OOT_BLACKLIST
    → 通过 MRO 查找 bridge 函数（RMSNorm → rms_norm_bridge）
    → 返回 bridge 函数作为 forward 方法
  → SGLang 使用框架参数调用 bridge 函数：
      rms_norm_bridge(self, x, residual, post_residual_addition)
    → Bridge 处理 SGLang 特有参数（post_residual_addition → 合并到 residual）
    → Bridge 调用 dispatch.call_op("rms_norm", obj, x, residual)
      → OpManager 通过 policy 解析最佳实现（flagos > vendor > reference）
      → 调用选中的后端：rms_norm_flaggems(obj, x, residual)
```

Bridge 层将框架特有参数与标准化算子签名解耦。厂商后端只需实现标准签名——同一份 impl 可在 sglang-plugin-FL 和 vllm-plugin-FL 上使用。

### Dispatch 架构（与 vllm-plugin-FL 共享）

```
┌─────────────────────────────────────────────────────────────┐
│  SGLang AROUND Hook        │  vLLM forward_oot override     │
│  (bridge/rms_norm.py)      │  (vllm_fl/ops/layernorm.py)    │
└────────────┬───────────────┴────────────────┬───────────────┘
             │                                │
             ▼                                ▼
┌─────────────────────────────────────────────────────────────┐
│  dispatch.call_op("rms_norm", obj, x, residual)             │
│  OpManager → SelectionPolicy → OpRegistry → resolve impl    │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌─────────────┐  ┌───────────┐  ┌──────────────┐
   │ DEFAULT     │  │ VENDOR    │  │ REFERENCE    │
   │ (FlagGems)  │  │ (Ascend/  │  │ (PyTorch)    │
   │ priority=150│  │  CUDA)    │  │ priority=50  │
   │             │  │ priority= │  │              │
   │             │  │   100     │  │              │
   └─────────────┘  └───────────┘  └──────────────┘
```

芯片厂商为两个框架实现**相同的后端接口**。唯一的框架特定代码是 bridge 层，由插件维护。

### ATen 替换

```
插件加载 → flag_gems.enable(record=True)
  → PyTorch dispatch table 为 ATen 算子注册 Triton kernels
  → 第一次推理调用时，每个被替换的算子会被记录
  → _AtenOnlyFilter 确保只记录 flag_gems.ops.* 命名空间的调用
    （排除 Layer 2 flagos 实现触发的内部 FlagGems 调用）
```

## 已知问题

- **不支持 Piecewise CUDA Graph**：FlagGems Triton kernels 包含 `logging.Logger` 调用，与 `torch.compile`（SGLang piecewise CUDA graph 使用）不兼容。启动服务时使用 `--disable-piecewise-cuda-graph`。常规 CUDA graph capture 正常工作。

## 许可证

Apache-2.0
