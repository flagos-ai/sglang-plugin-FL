# SGLang v0.5.11 → v0.5.18 升级梳理

本文只整理与 `sglang-plugin-FL` 适配直接相关的变化。当前实施顺序为
NVIDIA CUDA → Moore Threads MUSA → Huawei Ascend；三套环境不会同时抬升版本。

## 分版本变化

| 版本 | 主要变化 | 对插件的影响 |
|---|---|---|
| v0.5.11 | CUDA 13 / PyTorch 2.11 基线；旧 `MultiPlatformOp` 分发和单体 `CudaGraphRunner` | 插件原始基线 |
| v0.5.12 | 补齐 MUSA kernel 构建源码；平台和融合算子接口仍与 v0.5.11 基本一致 | MUSA 暂时保留在此版本 |
| v0.5.13 | NSA 命名统一为 DSA，`get_nsa_kv_pool_cls` / `NSATokenToKVPool` 政名 | 平台插件需同时提供新旧方法兼容 |
| v0.5.14 | CUDA Graph runner 按阶段拆分，decode runner 移到 `model_executor.runner` | 旧 `cuda_graph_runner.CudaGraphRunner` 导入失效 |
| v0.5.15 | 引入 `RuntimeContext`，Breakable CUDA Graph 进入默认执行体系 | 运行时配置和 graph 能力声明需重新验证 |
| v0.5.16 | 引入统一 `BaseFusedOp`；FLA kernel 移到 `sglang.kernels.ops`；平台接口增加 `device` 参数；混合模型默认使用 UnifiedRadixTree | 旧 `dispatch_forward` hook 和旧 FLA 路径不再可靠 |
| v0.5.17 | `ServerArgs` 在解析后只读；DP Attention 的 prefill graph 路径继续调整 | 插件只能在平台 defaults 生命周期内修改参数 |
| v0.5.18 | 融合算子全面迁移到 `BaseFusedOp`；`MultiPlatformOp` 仅保留兼容 shim；CUDA 依赖整体升级 | 必须使用 `register_oot_forward`，并重建 CUDA 镜像 |

## v0.5.18 NVIDIA 依赖基线

| 组件 | 版本 |
|---|---|
| Python | >= 3.10（本项目 CUDA 镜像使用 3.12） |
| PyTorch | 2.13.0 |
| torchvision | 0.28.0 |
| torchaudio | 2.11.0 |
| sglang-kernel | 0.4.6.post1 |
| FlashInfer | 0.6.17 |
| transformers | 5.12.1 |
| xgrammar | 0.2.1 |
| NVIDIA CUTLASS DSL | 4.6.2 |
| Triton 包 | 卸载（官方镜像原为 3.7.1） |
| FlagTree | 0.6.2a1（提供 Triton 3.6 模块） |
| FlagGems | master @ `8ea592557659491930ebf24c24392f958b29ac21` |
| SQLAlchemy | 2.0.48（FlagGems 运行依赖） |

### FlagTree / FlagGems 组合说明

- 先安装 PyTorch/SGLang，再彻底卸载 `triton` 包，最后安装
  `flagtree===0.6.2a1`。此时 `import triton` 来自 FlagTree，报告的模块版本为
  3.6.0；PyTorch 仍为 2.13.0+cu130。
- 当前 FlagGems master 源码会导入 `flag_gems.fused.DSA`，但该目录缺少
  `__init__.py`；由于上游使用 `find_packages()`，直接构建的 wheel 会漏掉
  DSA 模块并在导入时失败。
- CUDA 容器在临时虚拟环境中补齐 DSA package marker 并生成 wheel，再以
  `--no-deps` 安装，避免改写官方 SGLang/PyTorch 依赖栈；运行环境只补充
  FlagGems 缺失的 SQLAlchemy 依赖。
- 在 H100 上，FlagTree 0.6.2a1 + 该 FlagGems master 快照的 `_to_copy` 与
  `add` 均已通过 GPU 正确性冒烟，因此 NVIDIA 配置不再黑名单 `to_copy`。

## 插件改造清单

### NVIDIA CUDA（本轮）

- 使用 `BaseFusedOp.register_oot_forward` 注册 FL bridge。
- NVIDIA dispatch key 使用 `cuda`：已注册算子优先走 FL bridge，其他算子回落到
  SGLang 原生 `forward_cuda`，避免误退化到纯 PyTorch。
- 兼容新旧 DSA/NSA KV pool API。
- 兼容拆分后的 `DecodeCudaGraphRunner`。
- 兼容 `is_pin_memory_available(device=None)` 新签名。
- 同时适配新旧 FLA 模块路径。
- CUDA 容器及 CUDA CI 环境升级到 SGLang v0.5.18 依赖组合。
- 验证单卡/多卡推理、融合算子、CUDA Graph、NCCL/FlagCX 通信。

### MUSA（后续）

- 当前继续使用 SGLang v0.5.12、PyTorch/torch_musa 2.7.1。
- 需要单独解决 v0.5.16 以后平台发现阶段的 import cycle、MUSA kernel 构建和
  graph runner 变化，不能直接复用 CUDA 镜像升级方案。

### Ascend（后续）

- 当前继续使用 SGLang v0.5.11 的 CANN 8.5.0 A3 镜像。
- 需要对齐新版 `sgl-kernel-npu`、NPUGraphRunner、attention backend 和 DSA
  memory pool，再决定最终目标版本。

## 风险与验收门槛

1. `BaseFusedOp` 的 OOT registry 按具体 class 查找，RoPE 等已加载派生类也必须注册。
2. 未被 FL 覆盖的 NVIDIA 算子必须仍选择 `forward_cuda`。
3. Breakable/Piecewise CUDA Graph 与 FlagGems Triton kernel 的组合需做真实 GPU
   capture/replay，不能只靠 import 测试。
4. v0.5.18 的 PD disaggregation staging 元数据协议已扩展；FlagCX connector 需要
   独立的协议兼容测试。
5. 完成标准是：本地静态检查与单测通过，NVIDIA GPU 上 import、算子、单卡服务、
   TP 多卡和 CUDA Graph 冒烟全部通过。

## H100 阶段验证记录（2026-09-03）

- 基础镜像：
  `swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/lmsysorg/sglang:v0.5.18-runtime`
  （manifest digest `sha256:a667eab87d09e591529802266d63b583b721b851a9eeecbf8b05849c6f3929a7`）。
- 实测环境：NVIDIA H100 80GB HBM3，驱动 580.105.08；镜像内
  SGLang 0.5.18、PyTorch 2.13.0+cu130、CUDA 13.0、sglang-kernel
  0.4.6.post1、FlashInfer 0.6.17、transformers 5.12.1、xgrammar 0.2.1。
- 卸载 Triton 3.7.1 后安装 FlagTree 0.6.2a1；`import triton` 报告 3.6.0，
  实际路径来自 FlagTree 安装目录。Torch 仍为 2.13.0+cu130。
- FlagGems 使用 master commit
  `8ea592557659491930ebf24c24392f958b29ac21`（构建版本
  `5.3.6.dev0+g8ea5925`），wheel 只补充 DSA package marker；运行时保持
  `numpy 2.3.5` 和 `setuptools 84.0.0`。
- 自动平台发现成功：`PlatformFL`、vendor `nvidia`、device `cuda`、dispatch
  key `cuda`。
- 单元测试：195 passed；CUDA Graph 功能测试：5 passed；融合算子正确性测试：
  8 passed、3 skipped（按硬件能力跳过）。
- Layer 1：FlagGems 接管的 `to_copy` 与 `add` GPU 正确性冒烟均通过。
- Layer 2：RMSNorm 实际解析到 `sglang_fl.dispatch.bridge.rms_norm`，与 PyTorch
  参考实现 `allclose=True`，该次冒烟最大绝对误差为 0。
- Layer 4：补齐 v0.5.18 staging 类型兼容后，FlagCX PD backend 注册成功；尚未
  运行真实双实例 KV 传输。
- 尚未运行单卡服务、TP 多卡和 FlagCX 真实传输：8 张 GPU 当时均已有约 62–66GB 显存
  占用，共享目录也没有可在剩余显存内安全加载的小模型；应在获得空闲卡或小模型
  权重后继续。
