# 天数 CoreX：SGLang 0.5.18 empty 适配

以《天数智芯 empty使用文档》的 BI-V150 环境为基础，保留 CoreX 驱动、
厂商 PyTorch、FlagTree 和 FlagCX，升级 SGLang 到 **0.5.18**。
当前插件基线为 `9dc518f478`，已包含合并的
[天数原生适配 #132](https://github.com/flagos-ai/sglang-plugin-FL/pull/132)；
保留其中的独立 GDN 和 MoE 实现作为默认融合算子路径。
镜像、环境检查和验收组织方式同时参考
[Ascend #131](https://github.com/flagos-ai/sglang-plugin-FL/pull/131) 和
[MUSA #95](https://github.com/flagos-ai/sglang-plugin-FL/pull/95)。

| 组件 | 目标 |
| --- | --- |
| GPU | BI-V150，单卡 32 GiB；完整模型验收目标为 TP4 |
| Driver / CoreX | 4.5.0 / 4.5.0.20260629，沿用基线镜像 |
| PyTorch | 2.10.0+corex.4.5.0.20260629，沿用厂商版本 |
| FlagTree | 0.6.2a3+iluvatar3.6，提供 `triton` 模块 |
| FlagGems | 5.3.4，`f7c55cb2fe1fb90fc713eafa6f63d7cbb73453a9` |
| FlagCX | 0.13.0，沿用基线镜像 |
| SGLang | 0.5.18，`pyproject_other.toml` 的 `srt_empty` extra |

FlagGems 按使用文档保留源码 editable 安装。实机检查确认该版本普通 wheel
未包含 `flag_gems.fused.DSA`，会在插件初始化时失败；镜像配方使用
`pip install -e /opt/FlagGems` 避免这一打包缺口。

0.5.18 已包含 `srt_empty`，无需再应用旧文档的 0.5.11 empty patch。
不安装 NVIDIA Triton、sgl-kernel、FlashInfer、FlashAttention 或 ixformer。
独立的 [导入兼容包](../../addon/iluvatar-compat/README.md) 只提供上游
0.5.18 的有限导入符号；误入没有实现的 kernel 路径会明确报错。
它只在本平台显式安装，不是主插件的依赖。

## 构建独立镜像

从仓库根目录运行；保留原来的 0.5.11 容器便于对比。远程的构建、安装、
下载和测试任务均使用 tmux，防止 SSH 断开中止任务。

```bash
tmux new-session -d -s iluvatar-build \
  'docker build --target test -f docker/iluvatar/empty-0.5.18.containerfile -t sglang-fl:0.5.18-iluvatar . > /tmp/iluvatar-build.log 2>&1'
tmux capture-pane -t iluvatar-build -p
tail -n 60 /tmp/iluvatar-build.log
```

```bash
docker run -itd --name sglang-fl-iluvatar-0518 \
  --privileged --ipc=host --network=host --shm-size=128g \
  -v /mnt/share/models:/models \
  sglang-fl:0.5.18-iluvatar sleep infinity
docker exec -it sglang-fl-iluvatar-0518 bash
```

在已有隔离环境中安装时，先用上面的构建文件核对版本与依赖；以下安装命令在已建立的 tmux 会话内运行：

```bash
python -m pip install --no-deps ./addon/iluvatar-compat
python -m pip install --no-deps .
export SGLANG_PLATFORM=sglang_fl
export SGLANG_PLUGINS=sglang_fl
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_IS_FLASHINFER_AVAILABLE=0
export FLAGCX_PATH=/sgl-workspace/FlagCX
export SGLANG_FL_DIST_BACKEND=flagcx
export USE_FLAGGEMS=1

# 当前 211 隔离容器中的小算子检查只使用物理 GPU 15。
export CUDA_VISIBLE_DEVICES=15
python scripts/iluvatar/verify_environment.py --min-devices 1 --fused-profile vendor
```

`SGLANG_IS_FLASHINFER_AVAILABLE=0` 必须在启动 Python 前设置，确保 spawn
子进程不会因兼容包的存在而启用 FlashInfer。插件默认将 attention、sampling、
vision attention 分别设为 `triton`、`pytorch`、`triton_attn`，关闭 NVIDIA
custom all-reduce 和 piecewise graph。常规 decode graph 不被强制关闭，
其结果仍需独立验收。BF16 支持通过厂商 Torch 查询，显式 FP16/FP32
不受 NVIDIA SM 门槛影响。

SGLang 0.5.18 的 decode graph 批量上限字段为
`cuda_graph_max_bs_decode`，替代旧的 `cuda_graph_max_bs`。
当前例子使用新字段；单机 offline 和 concurrent 例子将
`DISABLE_CUDA_GRAPH=1/0` 真实传入引擎，分别用于关闭和开启 decode graph 的验收。

## 算子配置与状态语义

`sglang_fl/dispatch/config/iluvatar.yaml` 自动加载天数配置：

- ATen 黑名单为 `max`、`masked_scatter_`。
- 仅禁用 `vendor.cuda`，保留 `vendor.iluvatar`。
- Gemma RMSNorm 使用 `reference`。
- chunk、recurrent、packed decode GDN 和 Fused MoE 优先使用
  PR #132 的 `vendor.iluvatar`，其余融合算子按 FlagOS 优先策略选择。

无需复制旧文档的长环境变量串；旧配置中的
`SGLANG_FL_DENY_VENDORS=cuda,iluvatar` 会移除当前默认的原生路径。
手工覆盖时路由算子名是 **`topk`**，不是旧文档中的 `top_k`。

本次保留 PR #132 的独立原生计算，并对齐 SGLang 0.5.18 的公共契约：
state 使用 `[V,K]`；public recurrent 保持输入 state 不变，返回 FP32
final state；chunk 返回 `(output, None, h)`，`h` 为每 64-token chunk
起始状态，形状精确匹配上游；packed decode 原地写入调用者提供的
cache 和 `out`，负 cache 槽的输出遵循上游写零语义。

另提供 FlagGems 5.3.4 的 chunk、recurrent、packed decode 和 MoE 实现。
验收脚本默认 `FUSED_PROFILE=vendor`；设置 `FUSED_PROFILE=flagos` 时，
脚本通过 `SGLANG_FL_PER_OP` 将上述四类融合算子切换到 FlagOS，Gemma
RMSNorm 仍使用 reference。未量化标准 gated SiLU 是 MoE 的支持范围；
量化、EP 映射或特殊激活配置在执行前明确拒绝。
环境检查的 `--fused-profile vendor|flagos` 核对 `resolved` 中的实际路由，
不会自行改变算子配置。

PR #132 的已有验证属于其原生实现；它不能替代本次 FlagGems 路径的精度、
graph 或模型验收。两条路径需要分别保存版本、路由和测试结果。

CoreX 运行补丁还将位置 clamp 切换到 PyTorch，替换 NVIDIA legacy SM dtype
门槛，并补齐旧 Triton 扫描禁用 PDL 分支时需要的符号。真实 PDL 调用明确失败。

## 验收

先执行环境检查，再运行专门的算子精度及 graph 测试。这些测试在 CoreX 上
遇到缺失 kernel、数值错误或 graph capture/replay 错误会失败，
不会作为“可选能力”跳过。GPU 测试分别覆盖原生与 FlagGems 的支持范围。

```bash
tmux new-session -d -s iluvatar-ops \
  'python -m pytest -q tests/unit_tests/platform/test_iluvatar*.py tests/unit_tests/dispatch/backends/test_iluvatar_fla.py tests/unit_tests/dispatch/backends/test_iluvatar_native_contracts.py tests/unit_tests/dispatch/backends/test_flagos_empty_ops.py tests/functional_tests/ops/test_iluvatar_empty_ops.py > /tmp/iluvatar-ops.log 2>&1'
tmux capture-pane -t iluvatar-ops -p
tail -n 60 /tmp/iluvatar-ops.log
```

以下完整验收需四张显存充足的 GPU。脚本先以原生 FlagCX 执行多卡
all-reduce/all-gather，拒绝回退，再分别运行 Dense/MoE 的 graph 开关两种
offline 配置，以及 text、image、mixed 并发测试。可通过
`DENSE_MODEL_PATH`、`MOE_MODEL_PATH`、`TP_SIZE`、`CONCURRENT_N`、
`OUTPUT_DIR` 指定模型和日志目录。`CUDA_VISIBLE_DEVICES` 按实际分配的设备设置。

```bash
tmux new-session -d -s iluvatar-acceptance \
  'FUSED_PROFILE=vendor bash scripts/iluvatar/run_single_node_acceptance.sh > /tmp/iluvatar-acceptance.log 2>&1'
tmux capture-pane -t iluvatar-acceptance -p
```

原生验收结束并释放设备后，再验收 FlagGems 融合算子路径，保存独立日志：

```bash
tmux new-session -d -s iluvatar-acceptance-flagos \
  'FUSED_PROFILE=flagos bash scripts/iluvatar/run_single_node_acceptance.sh > /tmp/iluvatar-acceptance-flagos.log 2>&1'
```

统一测试入口也提供 BI-V150 配置（模型挂载路径 `/models`）：

```bash
tmux new-session -d -s iluvatar-e2e \
  'python tests/run.py --platform iluvatar --device bi_v150 --scope e2e > /tmp/iluvatar-e2e.log 2>&1'
```

两机 TP4+PP2 沿用 `examples/qwen3_6_27b_multinode.py` 和
`examples/qwen3_6_35b_a3b_multinode.py`，两边设置相同的软件栈与上述环境，
分别传 `--role master/worker --master-addr <主节点IP> --tp 4 --pp 2`；两机例子、服务与压测也通过 tmux 执行。
固定负载性能对比沿用旧文档的 64 请求、并发 64、输入 1K/4K/16K、输出
1K；服务启动后运行 `benchmarks/benchmark_throughput_serve.py --model <模型路径>`，
保留每次原始结果。共享 benchmark smoke 依赖额外的 Qwen3-0.6B 模型，因此
天数平台配置暂不启用该 smoke。

## 当前实机状态（2026-10-14）

已通过 JumpServer 的 `lxwang → u211 → new-root` 交互选择连接到 211。
该机 16 张 BI-V150 均运行已有 vLLM 任务，每卡约占用 24 GiB / 32 GiB。
本次另建隔离容器 `codex-iluvatar-0518-20261014`，仅设置
`CUDA_VISIBLE_DEVICES=15` 执行小算子测试，仍与 GPU 15 上的现有任务共享显存。

隔离容器已安装 SGLang 0.5.18、CoreX Torch 2.10.0、FlagGems 5.3.4 和
FlagTree 0.6.2a3。原生和 FlagGems 两条路径合计 **30 项 GPU 测试通过**，
覆盖 GDN 精度、64-token chunk 状态、变长和负槽语义、输入 state 所有权、
packed decode 缓存更新与 CUDA Graph 重放，以及 MoE bias、路由权重和缩放。
两种 profile 的完整运行时导入和实际路由检查均通过；另有 **12 项运行时测试**
通过，包含 CoreX 上 PDL 禁用分支的真实 Triton 编译及误启分支拒绝。
已有分布式 backend 映射的两项回归在先导入 SGLang 后通过；直接先导入
`sglang_fl.platform` 的循环导入是 PR #132 基线已有的顺序约束。

双模型 TP4、FlagCX 多卡、两机 PP 和吞吐尚未验收。当前各卡已有 vLLM
作业，需分配空闲设备后继续；单卡小算子通过不能替代模型和性能验收。
容器配方已提供，本次测试使用缓存厂商镜像派生的隔离环境手动安装同版本
软件栈，尚未实际构建、发布或启用新的天数 CI 镜像。
