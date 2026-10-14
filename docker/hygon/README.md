# 海光 BW1000：SGLang 0.5.18 empty 环境运行说明

[容器配方](empty-0.5.18.containerfile)保留原镜像的 DTK Torch、FlagTree、
FlagGems 和 native FlagCX。运行前需要设置以下环境变量；在宿主机直跑和进入
容器后执行 Python 时都应检查这些设置。

## 构建与启动

在海光宿主机的仓库根目录构建。安装、构建和模型验收耗时较长，应在 `tmux`
会话中运行。可以通过 `PIP_INDEX_URL` 指定可用的包镜像；不要设置全局 HTTP
代理。容器配方会保存并核对基底设备栈版本与 FlagTree 编译器文件 SHA256，
依赖冲突时构建失败，避免覆盖厂商 Torch 或安装独立 Triton。

```bash
tmux new -s hygon-upgrade

docker build --network=host \
  -f docker/hygon/empty-0.5.18.containerfile \
  -t sglang-fl:hygon-0.5.18 .

# 仅使用实际分配的空闲设备。此处以 2、3 号卡为例。
docker run -d --name hygon-sglang-0518 \
  --network=host --ipc=host --privileged \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /public-flash/models:/models:ro \
  -e HIP_VISIBLE_DEVICES=2,3 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  --entrypoint /usr/bin/env sglang-fl:hygon-0.5.18 \
  -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES sleep infinity

docker exec -it hygon-sglang-0518 bash
```

清除设备筛选变量的操作只作用于上面的容器主进程。每次 `docker exec` 执行
Python 时也应检查环境，或使用下面的形式显式清除：

```bash
docker exec hygon-sglang-0518 /usr/bin/env \
  -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
  python .github/scripts/hygon/verify_environment.py \
    --plugin-root /opt/sglang-plugin-fl \
    --baseline /opt/hygon-vendor-baseline.json --require-gpu --min-gpus 2
```

已在原始 empty 基底上保留并验证的版本：

| 组件 | 海光版本 |
|---|---|
| Python | 3.10.12 |
| SGLang | 0.5.18（应用本目录 HCU 源码补丁） |
| Torch | 2.10.0+das.opt1.dtk2604.2606151054.g98b1a8 |
| FlagTree | 0.6.1+hcu3.6 |
| FlagGems | 5.4.0.dev0 |
| sglang-kernel | 0.4.2.post2+das.dtk2604.torch2100.2607100943.gd00664 |
| flash-attn | 2.8.3+das.opt1.dtk2604.torch2100.2606131113.gebb4fa |

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

## 已验证范围（2026-10-14）

SGLang v0.5.18 tag 对应 `71de97b264b04dcd514cf904003028aefe9775c8`。
本次验收保留原始 empty 镜像的 DTK、FlagTree、FlagGems、厂商 kernel 和
native FlagCX；7 个厂商 wheel 版本及 FlagTree 三个编译器文件 SHA256 均一致。

| 验收项 | 源码 | 范围与结果 |
|---|---|---|
| 插件与环境回归 | `32f4411` | 本地 183 项插件测试、32 项验证器测试及 5 个子检查通过；另 5 项依赖实际 SGLang 的 API 用例在 HCU 环境通过 |
| 两模型完整 TP2/PP1 矩阵 | `8029ba2` | Qwen3.6-27B、Qwen3.6-35B-A3B 各在 eager 与 full decode graph 下通过 inference、serving、concurrent，共 12/12 轮；包含文本、4 项图片语义、HTTP 插件激活/分发和 text/VL/mixed 各 16 请求并发 |
| 最终镜像复验 | `32f4411` | GPU 环境检查、27B graph 的 5020-token 长输入及 27B/35B graph 两题文本均通过，任务退出码为 0 |
| 原生 FlagCX | `8029ba2` / `32f4411` | 单机 2 rank，以及最终版本物理双机 4 rank 的 FP32/BF16 AllReduce、FP32 AllGather 误差均为 0，禁止回退，显式释放 communicator |
| 页表专项 | `a6706de` | 真实 GPU 对 4097/8192 长度、非顺序请求索引、尾部清理和稳定 int32 缓冲通过逐元素校验 |
| 单机 TP1/PP2 | 27B `a6706de`；35B `8029ba2` | 两模型均通过两题文本和 Engine shutdown |
| 最终版本物理双节点 TP2/PP2 | `32f4411` | 两模型各通过 54 个语义请求（2 text、4 VL、text/VL/mixed 各 16 并发）；四 rank 同步并自然退出，两端原始进程退出码为 0，四卡显存均恢复约 99.5% 空闲 |

`a6706de` 与 `8029ba2` 的 attention、FLA、MoE 实现相同；后者补充了 CLI
解析前的 HCU 注册。最终运行时代码 `32f4411` 另外修复 PP 退出时尚未完成的
通信：先排空流水线和通信队列，同步 GPU 与全部 scheduler，再执行退出。
首次双机 27B 的 54 个语义请求虽通过，但从节点退出码为 137，故该次整体验收
判为失败，不能作为自然退出通过的证据。

完整矩阵的共同父镜像为 `codex-hygon-sglang:0.5.18-8029ba2`，ID：
`sha256:f514c23b69f9e7249035979e7921c6e741371c81bb8ec7447f318d834e107063`。
最终版本在三台测试机上从该父镜像及同一份 LF 源码归档派生，tag 均为
`codex-hygon-sglang:0.5.18-32f4411`。三次构建时间不同，image ID 分别为：

| SSH 别名 | 最终 image ID |
|---|---|
| `hy_10` | `sha256:284fa4821b36a719deb228d3429ef3d703807748b800925d1baae12ed0fb2947` |
| `hy_11` | `sha256:4fc739c4ed8473b579a49fba79cf547882eb5843d8f4cc4731c446e2b9a48775` |
| `hy_12` | `sha256:bced5ccadb2f788328e34d3514108d510bcf452fd93da3a8cf1507a4939e39f4` |

这些是测试机上的本地镜像。派生源码归档 SHA256 为
`275e9a0524aea5da4215bfb242f8962dce16be421f36c13966584970f4a53fcd`；
部署到其它机器可按前面的容器配方重新构建。没有将本地镜像推送到 registry。

物理双机测试使用 `hy_11` 和 `hy_12` 各自分配的 2、3 号卡、`--uts=host`，
以及绑定实际网卡 `bond1` 的 Gloo/FlagCX。日志显示跨机数据走 `NET/Socket`，
机内走 `P2P/IPC`。双节点验收额外要求发送真实 `ShutdownReq` 后，四个
scheduler 自然退出、本地进程退出码均为 0、从节点原始 CLI 退出码为 0，
且清理容器前显存已释放；同时拒绝含 scheduler 异常或 SIGQUIT 的日志。

本次双节点 TP2/PP2 使用 eager 模式；没有验证该拓扑的 graph 或 RDMA。
未执行文档中 1K/4K/16K 输入、1K 输出、64 请求的 empty/non-empty 性能对照。
