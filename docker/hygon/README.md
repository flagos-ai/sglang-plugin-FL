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
[`sglang-v0.5.18-hcu.patch`](../tools/wheels/hygon/sglang-v0.5.18-hcu.patch)，然后再安装 SGLang。
仅安装或更新插件无法修复这一上游导入问题：量化注册表在插件补丁执行前就会
根据 HIP 标记导入 AMD Quark/AITER 内核；HCU 同样使用 HIP，但不提供这些内核。

容器配方负责这一安装顺序。手动从源码部署时，在安装 SGLang 前执行：

```bash
export PLUGIN_ROOT=/opt/sglang-plugin-fl
export SGLANG_SOURCE_ROOT=/opt/sglang-0.5.18
cd "$SGLANG_SOURCE_ROOT"
git apply --check "$PLUGIN_ROOT/tools/wheels/hygon/sglang-v0.5.18-hcu.patch"
git apply "$PLUGIN_ROOT/tools/wheels/hygon/sglang-v0.5.18-hcu.patch"
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

## MTP：Qwen3.6-27B / 35B-A3B（单机 eager 已验收）

本节对应 [PR #149](https://github.com/flagos-ai/sglang-plugin-FL/pull/149) 的单机海光 MTP 实现。
已验运行时代码为 `c46d2258b2362f0ea171b30781683138f8c8de1b`；PR 当前尚未合并。
两款正式 BF16 模型在下述 eager 配置完成了批量与 MTP/baseline 文本回归。
设备栈沿用原镜像，插件需要使用包含本次改动的源码。在已准备的容器内取得该版本：

```bash
git clone --no-checkout https://github.com/flagos-ai/sglang-plugin-FL.git sglang-plugin-FL-mtp
cd sglang-plugin-FL-mtp
git fetch origin refs/pull/149/head
git checkout --detach c46d2258b2362f0ea171b30781683138f8c8de1b
```

从该 checkout 按上文配方构建；已发布的
`0.5.18-hygon-ci-20261015` 镜像提供设备栈，需要另行安装当前插件源码。
沿用该镜像时，在 `docker run` 中挂载当前仓库
`-v "$PWD":/opt/sglang-plugin-fl`，并在容器内安装，保留厂商依赖：

```bash
cd /opt/sglang-plugin-fl
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python -m pip install --no-deps --no-build-isolation -e .
```

两模型均需包含完整 MTP head 的正式 BF16 checkpoint；配置中的 MTP 字段不能
代替实际权重。插件会检查真实 loader 加载的参数及融合分片，缺失即失败。
模型分别只读挂载到 `/models/Qwen3.6-27B` 与 `/models/Qwen3.6-35B-A3B`。
例如，35B 位于 `/public-flash/lishuaiao/models-hf-bf16/Qwen3.6-35B-A3B` 时，
在 `docker run` 中另加
`-v /public-flash/lishuaiao/models-hf-bf16/Qwen3.6-35B-A3B:/models/Qwen3.6-35B-A3B:ro`。

在宿主机 `tmux` 会话中使用已准备的容器，先完成上文 GPU 环境检查，再逐模型
执行完整示例。`HIP_VISIBLE_DEVICES` 必须是实际分配的两张空闲卡；保留每次日志
和原始退出码。

```bash
set -euo pipefail
cd /opt/sglang-plugin-fl
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
: "${HIP_VISIBLE_DEVICES:?先指定已分配的两张海光卡}"
export SGLANG_PLUGINS=sglang_fl
export SGLANG_FL_DIST_BACKEND=flagcx
export FLAGCX_PATH=/sgl-workspace/FlagCX

MODEL_PATH=/models/Qwen3.6-27B TP_SIZE=2 \
  python examples/qwen3_6_27b_mtp_inference.py \
    --disable-cuda-graph --disable-piecewise-cuda-graph \
    --disable-overlap-schedule

MODEL_PATH=/models/Qwen3.6-35B-A3B TP_SIZE=2 \
  python examples/qwen3_6_27b_mtp_inference.py \
    --disable-cuda-graph --disable-piecewise-cuda-graph \
    --disable-overlap-schedule
```

受控配置为单机 TP2/PP1、BF16 模型及 KV、HCU attention、同步 eager 执行，
使用 `EAGLE`、3 steps、topk=1、4 draft tokens，同一 checkpoint 提供 MTP head。
这里的 3 steps 不是 `EAGLE3` 算法。示例固定 page64、context4096、
`mem_fraction_static=0.70`、`max_running_requests=16`、`chunked_prefill_size=4096`，
关闭 radix cache、两阶段 graph 与 overlap scheduling。每次创建 MTP 或 baseline
Engine 前都会重设 Torch 自动加载开关；启动 Python 前仍需设置上述环境。

当前实现明确拒绝 MTP graph、topk>1 分叉树和启用 overlap 的配置。
Engine B4 批量文本已验收；其它拓扑及 MTP 视觉、C64 高并发和 HTTP 入口尚未验收。这些边界来自当前软件实现与
验证范围。普通非 MTP decode graph 的既有使用方式保持不变。

完整验收保留 baseline 比较，要求语义契约、64-token teacher-forced 检查、
实际 speculative 计数和各 scheduler 接受长度严格大于 2 的检查全部通过，
并核对原始退出与资源清理。不要用 `--skip-baseline` 的局部结果、权重覆盖日志
或组件测试代替完整验收；示例的单请求 tok/s 仅用于诊断。

### 实机回归（2026-10-23）

| 检查 | Qwen3.6-27B | Qwen3.6-35B-A3B |
|---|---|---|
| MTP / baseline 完整文本示例 | 30 次调用、24 项语义检查通过 | 30 次调用、24 项语义检查通过 |
| 教师强制数值检查 | 64 token；最大 logprob 差 0.242519；近似并列 0、违规 0 | 64 token；最大 logprob 差 0.331164；近似并列 1、违规 0 |
| 完整示例 scheduler 窗口平均接受长度 | 2.9916667 | 2.8444444 |
| 批量 MTP | 4 批 × 4 请求，16 项语义、token、logprob、top5 与实际 speculative 计数通过 | 4 批 × 4 请求，16 项语义、token、logprob、top5 与实际 speculative 计数通过 |
| 独立统计探针 | 实际 256 输出 token、91 次 verify；不计入 16 条批量请求 | 实际 256 输出 token、100 次 verify；不计入 16 条批量请求 |

236 项海光单元测试与 19 项真实 GPU 数值测试均通过、无跳过。两模型的原始
驱动/docker-exec 退出码均为 0，完整响应和源码哈希经独立审计；本任务容器
停止前后已分配 GPU 均释放至约 2 MiB。scheduler 诊断和 resource_tracker
警告保留，不能据此宣称每个 rank 自然退出或没有 IPC 泄漏。

批量统计与独立探针分开计数，scheduler 窗口均值与整段 token/verify 比值分开记录。
全部生成请求核对了实际 output IDs 和长度；数值重评分范围为每模型 64 个
teacher-forced token。示例的单请求 tok/s 仅用于诊断，没有受控 MTP 加速比结论。

海光 speculative logprob 另修复了合法 `-1` padding 的高级索引语义，并复用上游
稳定行归一化计算，保留温度、输出 dtype、top-k 和指定 token 概率等行为。
原 GPU 输入和 `rtol=1e-5`、`atol=2e-5` 门槛保持不变。

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

## GitHub CI

海光复用 main 分支的 `ci-sglang-plugin-fl-hygon` runner、驱动挂载和共享
unit/functional/E2E 流程。CI 配置见 [hygon.yml](../../.github/configs/hygon.yml)；
模型目录从宿主机 `/data/models/Qwen` 只读映射到 `/models`。

已发布的 CI tag 为
`harbor.baai.ac.cn/plugin/sglang-plugin-fl:0.5.18-hygon-ci-20261015`，工作流固定为：

```text
harbor.baai.ac.cn/plugin/sglang-plugin-fl@sha256:d21327c133c4abd40297cf717278bd9190397916a544dc84cbfe9068f09b6c96
```

该镜像从已验证运行时派生，只增加 `pytest-timeout==2.4.0`，构建后重新核对
厂商包版本与编译器文件哈希；发布时已核对匿名 registry manifest 与实际镜像 ID。
复现构建先按上文生成 `sglang-fl:hygon-0.5.18`，再执行：

```bash
docker build -f docker/hygon/ci-0.5.18.containerfile -t sglang-fl:hygon-0.5.18-ci .
```

CI 在安装插件前拒绝旧的 SGLang 0.5.11 镜像。安装只使用
`--no-deps --no-build-isolation`，并核对基底版本、编译器哈希和真实 GPU 数值路径。
设备分配以 `HIP_VISIBLE_DEVICES` 为准；仅有 ROCR/CUDA 分配时保留其顺序转为
HIP，存在不一致的双重映射时明确失败。CI 不自动挑选或抢占其它 GPU。

E2E 保留 eager/graph × inference/serving/concurrent 的原六组，新增仅 eager 的
`mtp` 两模型组，共七组、十四个 model case，同平台组串行执行。MTP 组核对
真实 speculative counters、B4 输出和原 64-token teacher-force 门禁；graph
不选择 MTP。分配好设备后可单独执行：

```bash
python tests/run.py --platform hygon --device bw1000 --scope e2e --task mtp
```

benchmark 保持显式启用。上面的 MTP 结果来自手动实机验收；工作流配置完成和
镜像发布不代表 GitHub 硬件 CI 通过，应以对应 PR 最新提交的实际 job 结果为准。

## 两-wheel CI 镜像：构建与构建后验收

`runtime-0.5.18.containerfile` 只做镜像组装：Runtime 基底（moving semver
tag，见下）、一次 uv 解析安装两个真实 wheel（派生 HCU wheel
`sglang-0.5.18+flagos.hcu1`——由 `tools/wheels/hygon/derive_sglang_wheel.py` 在
staging 期从 SHA 钉定的 `sglang-0.5.18+flagos` 输入 wheel 折入 reviewed HCU
patch 而来——与本仓库构建的 `sglang_fl`）、以及 HCU/DTK 兼容层。所有验收都不在镜像构建
期执行：

```bash
# 组装构建 + 构建后验收 + 可选推送（唯一发布入口：验收是发布门，
# --push 只在验收通过后可达；staging + tag→digest 解析 + 从解析出的
# digest 动态采集 runtime-constraints + dry-run 计划门 + docker build
# 带 provenance label + verify_image.sh 八阶段验收）
docker/hygon/build_two_wheel.sh [--push] <tag>

# 也可对已构建镜像单独验收（基底从镜像自身的 OCI label 读取）
docker/hygon/scripts/verify_image.sh <tag>
```

- 基底是 Runtime owner 要求的 moving semver tag
  （`flagos-runtime-hygon-dtk26.04:2.3.0`；固定 digest 曾被 registry 清理
  而失效）。`build_two_wheel.sh` 构建前把 tag 解析成 repo digest：参考
  数据采集、dry-run 门、`docker build` 全部用同一 digest，并把 digest 记
  进 `org.flagos.ci.base-digest` label，验收从该 label 回读基底——tag 再
  移动也不影响已构建镜像的可复现验收；
- `datasets==5.2.0` 已按 Runtime owner 的 review 意见进入插件的 Hygon extra：
  Runtime 自带 fsspec 2026.9.0，平台 index 上的更旧 datasets 都对 fsspec 封顶
  而导致 ResolutionImpossible，5.2.0 是首个放开该上限的测试版本。Runtime
  动态 constraints 则由 `build_two_wheel.sh` 从解析出的 digest 生成，只保护
  Runtime distributions 不被 resolver 替换，仅作构建临时输入（/tmp，同一
  RUN 内消费并删除），不留在最终镜像；
- 构建输入（constraints、两个应用 wheel、修正元数据的 xgrammar wheel、
  excludes）由 BuildKit 只读 bind mount
  给唯一安装 `RUN`，不进入最终镜像的任何 blob；uv 同时使用 `--no-cache`，
  验收明确拒绝 `/root/.cache/uv`（旧候选曾残留约 1.1 GB）。镜像不设 `WORKDIR`
  （继承 Runtime 的 `/app`），不含 `/opt/sglang-ci` 或任何构建痕迹。HCU
  patch 不再进入镜像：它折入派生 wheel（staging 期严格差异门禁保证只有
  patch 文件、METADATA Version 行、dist-info 目录名与 RECORD 变化）；派生
  wheel 的 SHA256 记为 `org.flagos.ci.sglang-wheel-sha256` label，验收期核
  对已安装 distribution 版本即 `0.5.18+flagos.hcu1`。xgrammar 同样在 staging
  期从 SHA 钉定的官方 `0.2.1` wheel 派生为 `0.2.1+flagos.hcu1`：只删除 HCU
  上错误的 active `Requires-Dist: triton`，严格门保证其余 METADATA、tags、
  payload/native 文件逐字节不变；其 SHA 由
  `org.flagos.ci.xgrammar-wheel-sha256` 与外置 provenance 双重绑定；
- 参考数据（freeze、site-packages 哈希、pip-check 基线、provenance）每次
  从 label 指向的 digest 现场采集，只存在于 `mktemp`/CI artifact，不提交
  仓库，也不进镜像；
- `verify_image.sh` 覆盖：runtime 零漂移六门、vendor 模块 provenance、
  plugin guard 冒烟、Runtime clang 无卡 HSACO 编译、SGLang import 冒烟、Triton 多入口
  解析、installed-wheel 父/子进程解析、镜像级 ENV/label 合同（含
  LD_LIBRARY_PATH 前插语义、中性 WORKDIR、无 CI 工作目录残留）。Triton 相关
  各门按镜像 label `org.flagos.ci.triton-slot` 声明的 slot 取**该 slot 的真实
  合同**（见下文），没有为实验放宽的统一口径；
- wheel 的 METADATA/entry point/payload 验收在 staging 期
  （`scripts/check_plugin_wheel.py`）与验收期（六门之五）各跑一次。

### 候选输入：staging、内容钉定与发布边界

`docker/hygon/stage_wheels.sh` 产出构建上下文（wheel、provenance、constraints、
excludes；disposable，不入仓库、不进镜像）。一条版本只对应一份内容：

- 两个派生 wheel 的 SHA256 在脚本内钉定（`SGLANG_DERIVED_SHA256`、
  `XGRAMMAR_DERIVED_SHA256`）。现场派生结果与钉定值不符即失败：内容变了就要
  下一个 local version（`hcu2`），绝不重发 `hcu1`；
- 已派生好的 wheel（发布件，或有 SHA 佐证的本地副本）用
  `SGLANG_DERIVED_WHEEL` / `XGRAMMAR_DERIVED_WHEEL` 指定：staging 复制它与
  相邻的 provenance JSON，并比对同一钉定值，不在镜像内解包、patch、重打包；
- 插件 wheel `sglang_fl-0.1.0` 由 `scripts/stage_plugin_wheel.py` 进 staging，
  有两种不能混用的身份：
  - **开发候选**（默认）：从当前 checkout 构建，zip 时间戳取自提交
    （`SOURCE_DATE_EPOCH` = 提交时间），同一提交两次构建逐字节相同。此前的
    差异只在 6 个 dist-info 成员的时间戳上。可复现不等于可发布：这种镜像不能
    push；
  - **release 输入**：`SGLANG_FL_WHEEL=<已发布的 wheel>` 加
    `SGLANG_FL_SHA256=<其钉定 digest>`，wheel 旁边必须有记录了构建提交
    `source_commit` 的 provenance JSON。staging 只校验和复制，不构建；
  - `build_two_wheel.sh --push` 在插件 wheel 不是已发布制品时直接拒绝。镜像
    label `org.flagos.ci.sglang-fl-wheel-sha256` 记录它的 digest；严格验收逐个
    比较已安装的 `sglang_fl` 文件与 wheel 内容（installer 改写的 RECORD
    除外），并对其 provenance 做与派生 wheel 相同的谱系检查；
- staging 末尾写出 `publication-manifest.json`
  （`tools/wheels/hygon/publication_manifest.py`）：每个制品的 distribution、
  version、文件名、字节数、SHA256 与建议的 Nexus 仓库/路径
  （`flagos-pypi-hygon`，`packages/<name>/<version>/<file>`，与已发布输入
  wheel 的现有布局一致）。本轮全部制品状态为 LOCAL DEVELOPMENT CANDIDATE，
  没有发布任何内容；reviewer r4627443492 要求的“wheel 来自
  resource.flagos.net”在制品 owner 发布之前属于 externally blocked；
- 正式镜像只应消费已发布、不可变、SHA 钉定的 wheel。已发布件一定早于镜像的
  提交派生，所以严格验收对 provenance 的 `source_commit` 要求的是**谱系**而
  不是相等：派生提交必须存在于本仓库，且是镜像 `source-commit` 的祖先
  （`scripts/provenance_lineage.py`）。wheel 的内容绑定仍由 SHA256 label 与
  逐文件 manifest 负责，这部分没有放宽。

### 插件自有的 kernel import guard 与激活时序

HCU Runtime 没有 `sgl-kernel` / FlashInfer wheel，而 SGLang 0.5.18 在加载层模块时
会 eager import 它们。reviewer r4627477929 指出这些 stub 应属于 plugin repo：
它们现在位于 `sglang_fl` wheel 内
（`sglang_fl/dispatch/backends/vendor/hygon/early.py` + `import_guard.py`），
镜像里不再写任何 stub 文件或 `.pth`。

- **方案 A（early activation，已实现）**：Hygon 平台发现（`sglang_fl` 的 vendor
  early hook）执行时，把只对 `sgl_kernel`、`flashinfer` 两个根生效的 placeholder
  finder **追加**到 `sys.meta_path` 末尾——真实已安装的包永远优先；每个包可用
  `SGLANG_FL_HYGON_SGL_KERNEL_STUB=0` / `SGLANG_FL_HYGON_FLASHINFER_STUB=0`
  单独关闭。这不是 #197 那种全局 import-adjusting hook：finder 只回答这两个
  根下的名字；
- **语义与原镜像 stub 一致**：`sgl_kernel.utils.is_arch_support_pdl()` 返回
  `False`；`sgl_kernel.scalar_type` 永不伪造（`find_spec` 返回 `None`，保住
  SGLang 的纯 Python fallback）；其它对象可 import、调用即抛 `RuntimeError`；
  不伪造 distribution metadata；`SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` 只在
  `sgl_kernel` placeholder 生效时写入，与 stubs 同进退；
- **激活时序实证**（无卡；真实 `python -m sglang.launch_server` 冷启动，CPU
  torch 2.9.0 + 伪造 HCU 设备信息）：`sglang_fl.activate_platform()` 首次执行于
  `import sglang`（`sglang/__init__.py:21` → `sglang/srt/utils/__init__.py:2`
  → `sglang/srt/utils/common.py:100` 的 `current_platform` 惰性解析），此时仅
  19 个 sglang 模块已加载；激活之前 sgl_kernel/flashinfer 的 import 请求数为
  0，直到 launch import、`load_plugins()` 与 `prepare_server_args()` 结束也
  没有（guard 自身的 `_installed()` 探测除外）。对这 19 个模块的全分支
  module-scope import 闭包做静态复核，同样为 0。激活之后 placeholder 对两个
  根都生效：`is_arch_support_pdl()` 为 `False`，
  `find_spec("sgl_kernel.scalar_type")` 为 `None`；
- **卸载无残留**：guard 随 `sglang_fl` distribution 安装/卸载，没有 `.pth`、
  没有物理 `sgl_kernel`/`flashinfer` 包（单测
  `test_plugin_wheel_carries_the_guard_and_uninstall_leaves_nothing`）；
- **构建后验收按行为检查**：`verify_runtime_immutable.py` 在镜像内导入已安装
  的 guard 文件并断言上述语义，而不是匹配源码字符串。

局限：CPU torch 跑不了 `torch.cuda` 相关的 import 期代码，engine import 链在
“Torch not compiled with CUDA enabled”处停止；真实入口上的时序确认仍欠 Hygon
硬件矩阵。

### clang wrapper 替代矩阵（reviewer r4627367002）

Reviewer 不认可“塞进镜像的 clang 包装脚本”。候选改用**直接调用 Runtime 自带的
`/opt/dtk/aillvm/bin/clang-18`**，依据是下面的无卡真实编译探针；**wrapper 能否
最终删除，以 Hygon 硬件矩阵上的真实 kernel 编译为准，验证前不作此声明。**

探针：在解析出的 Runtime digest
`acccfccbde7681de6615ae2b411a9c642b14edbd3f4bdb74d75477028878a9f6` 的临时容器内，
调用 FlagTree HCU backend **自己的** `make_amdgcn` + `make_hsaco` 编译含
`__ocml_fabs_f32` 的验收 IR（`scripts/hcu-device-lib-check.ll`，gfx936），
`scripts/flagtree_compile_probe.py` 记录 clang 路径、argv、cc1 载入的 device
bitcode 和产物 SHA256。“旧布局”指模拟的更早 Runtime：没有 `/opt/dtk/amdgcn`，
device libs 只在 `/opt/dtk/dcc/dccgcn/bitcode`（build-infra #1786 的现场）。

| 变体 | 相对 A 的唯一差异 | 当前布局 | 旧布局 |
|---|---|---|---|
| **A 直接 Runtime clang（候选采用）** | —（FlagTree 默认参数，含 1 个空串参数） | 通过：asm `10a0c529…`，HSACO `ea37ba47…`（3040 B），device bitcode 取自 `/opt/dtk/amdgcn/bitcode` | **失败**：`clang-18: error: cannot find ROCm device library; provide its path via '--rocm-path' or '--rocm-device-lib-path', or pass '-nogpulib' to build without ROCm device library`（即 #1786） |
| B 已删除的 wrapper（对照基线） | `TRITON_HIP_CLANG_PATH` 指向包装脚本：丢弃空串参数并追加 `--rocm-device-lib-path=/opt/dtk/dcc/dccgcn/bitcode` | 通过，产物与 A 逐字节相同（bitcode 取自 `/opt/dtk/dcc/dccgcn/bitcode`） | 通过，产物与 A（当前布局）相同 |
| C 目录别名 | 新增 `/opt/dtk/amdgcn/bitcode → /opt/dtk/dcc/dccgcn/bitcode` 符号链接，不改任何 Runtime 文件 | 不适用（标准路径已存在） | 通过，产物相同 |
| D1 backend 追加 `-nogpulib` 并去掉空串参数 | FlagTree 源码改动（原型副本） | **失败**：asm 不同（`991c06b4…`，cc1 未载入任何 device bitcode），链接步 `ld.lld: error: undefined symbol: __ocml_fabs_f32` | 未跑 |
| D2 backend 显式传 `--rocm-device-lib-path=<backend 自算的 libdir>` 并去掉空串参数 | FlagTree 源码改动（原型副本） | 通过，产物相同 | 通过，产物相同 |

A 的 IR→assembly argv（当前布局）：

```text
clang -target amdgcn-amd-amdhsa -mcpu=gfx936:xnack- -mllvm=-check-valu-data-forward-hazards=0 \
  -mllvm=-disable-cluster-lds-memops=true -mllvm=-hcu-pre-emit-load-store-opt=false '' \
  -mllvm=-support-768-vgprs=true -mllvm=-enable-hcu-approx-fp-math=true \
  -mllvm=-hcu-update-wait-by-reverse-search=true -O3 <probe>.ll -S -o <out>.amdgcn
```

结论与去向：

- 当前 Runtime 上 A 成立，是因为 device libs 同时存在于 clang 自动探测的标准
  路径 `/opt/dtk/amdgcn/bitcode`；backend 把 device libs 链进 IR 之后，交给
  clang 的 IR 仍声明 `__ocml_*` / `__ockl_*`，clang 这一步自己也必须找到它们；
- A 对 Runtime 布局敏感（旧布局即 #1786）。若硬件矩阵或未来的 Runtime 上 A
  失败，不需要恢复 wrapper：C（目录别名）与 D2 都产出与 A 相同的产物；
- D 是根治方案，需要 FlagTree / FlagOS Runtime owner 落地：候选补丁在
  [`upstream/flagtree-0.7.0+hcu3.6-explicit-device-lib-dir.patch`](upstream/flagtree-0.7.0+hcu3.6-explicit-device-lib-dir.patch)，
  本仓库从不应用它；
- 构建后验收与 CI 环境门要求直接选中 Runtime clang，并拒绝
  `TRITON_HIP_CLANG_PATH` 等覆盖（探针负例已验证：旧布局、存在覆盖两种情况均
  FATAL）。

复跑探针（`verify_image.sh` 第 4 阶段即此命令；矩阵脚本本身属 disposable
证据，不提交）：

```bash
docker run --rm --platform linux/amd64 \
  --volume "$PWD/docker/hygon/scripts":/scripts:ro \
  --entrypoint python <candidate-image> /scripts/flagtree_compile_probe.py \
  --probe /scripts/hcu-device-lib-check.ll
```

### Triton 编译器 slot：`/opt/flagtree` vs `/opt/triton`（reviewer r4627164234）

Runtime 自带两套编译器树，容器文件只是用 `ENV PYTHONPATH` 选其一：

| slot | 模块版本 | triton dist-info | backends | pip check 是否看到 triton |
|---|---|---|---|---|
| `/opt/flagtree`（生产默认） | 3.6.0 | 无（由 `flagtree 0.7.0+hcu3.6` distribution 提供） | `hcu` | 否 |
| `/opt/triton` | 3.5.1 | `triton-3.5.1+das.opt1.dtk2604.torch290.dist-info` | `amd`、`nvidia` | 是 |

- 候选默认继续 `/opt/flagtree`。镜像 label `org.flagos.ci.triton-slot` 记录所选
  slot（构建脚本写入，值只允许 `/opt/flagtree` 或 `/opt/triton`），验收脚本从
  label 读回，并按**该 slot 的真实合同**逐项核对：模块版本、dist-info 有无、
  backend 目录、`import triton` 的解析路径、pip check 是否看到 triton。合同定义
  在 `scripts/triton_slot.py`，没有为 A/B 放宽任何一条；
- `/opt/triton` 下的 dist-info 让 `pip check` 即使对**原始** xgrammar wheel 也
  满足，但它把编译器换成 vendor 3.5.1（没有 `hcu` backend）；该组合在 HCU 上
  是否可用未知，只能由硬件实验回答。

A/B 命令（单变量；两臂都是同一个已验收候选之上的薄层）：

```bash
docker/hygon/triton_slot_ab.sh --experiment <ID> --context <staging-context-dir> \
  <已验收候选镜像> <tag-a> <tag-b>
# A 臂 PYTHONPATH=/opt/flagtree，B 臂 PYTHONPATH=/opt/triton
```

脚本先证明单变量（`scripts/verify_triton_slot_pair.py`：两臂 `RootFS.Layers`
相同、ENV 只差 `PYTHONPATH`、label 只差 slot label、`/opt/flagtree`、`/opt/triton`
与 site-packages 的树摘要相同、基底 digest 与两个 wheel 的 SHA label 相同），
再对每臂跑一遍完整 `verify_image.sh`，不推送。无卡部分只在一个伪候选上验证
过（31 层相同；B 臂报告 triton distribution
`3.5.1+das.opt1.dtk2604.torch290`）；**真实候选上的 A/B 尚未运行**。

硬件 A/B 与解除条件：

1. 候选镜像先通过无卡验收（需 Nexus 可用以完成构建）；
2. 用一次性分支把 `.github/configs/hygon.yml` 的 `ci_image` 指向 A 臂，另一
   分支指向 B 臂，各自推送镜像并跑 Hygon 硬件矩阵——本轮不允许改该文件、
   不允许推送，故未执行；
3. 只有 A、B 两臂在同一候选 digest 上各自跑完矩阵才形成 slot 结论；A 臂自身
   失败的实验无效。此前的硬件 A 尝试即属此类：基线 A 因 amdsmi×hyhal ABI
   失败，变量未隔离，结论无效；
4. 在此之前容器文件默认继续 `/opt/flagtree`。

### `--excludes triton` 的退役门

修正元数据的 xgrammar wheel 去掉了闭包里唯一活跃的 `triton` 依赖，因此过渡期的
`--excludes triton`（及记录它的 pip-check 精确豁免）在删除前必须先被证明已成
no-op。`docker/hygon/no_excludes_gate.sh` 对**真实索引**、同一解析 digest、同一份
staged artifacts 跑两次 `uv pip install --dry-run`：

- plan A：containerfile 的安装参数原样（由 `scripts/uv_install_contract.py` 从
  containerfile 提取，仍带 `--excludes`）；
- plan B：同一命令去掉 `--excludes <file>`。

两个 plan 都要过常规 plan 门（选中 `xgrammar==0.2.1+flagos.hcu1`、无 standalone
triton、零 Runtime distribution 被替换），且必须逐行相同，plan B 还必须选中
`datasets==5.2.0`。

```bash
docker/hygon/no_excludes_gate.sh <staging-context-dir> [--evidence-dir DIR]
```

删除 `runtime-0.5.18.excludes.txt` 与精确豁免的条件（缺一不可）：gate 通过 →
删除后的候选构建 → `pip check` → 完整 `verify_image.sh`。索引不可用时 gate 失败
并保留全部 resolver 输出；不改索引优先级、不伪造已安装 triton metadata 绕过。

### 剩余兼容债速查（Docker 注释保持简短，完整证据口径在此）

容器文件里的 TEMP 块共用一句头注释：“TEMP Runtime compatibility bundle；这些
设置暴露 Runtime 已携带但没有对所有入口稳定提供的能力；等待 Runtime owner 明确
迁移方案；详细 observed failure、owner 和 removal gate 见 docker/hygon/README.md。”
下表就是这张清单。CI 结论均按 job 级数据核对。

| 兼容块 | observed failure（首错误） | 证据 | owner | removal gate |
|---|---|---|---|---|
| `ENV PYTHONPATH=/opt/flagtree`（Triton slot 选择） | 无 bash profile 的直接 python 入口 `find_spec("triton") is None` | base 探针；slot 合同见上节 | FlagOS Runtime | Runtime 以 OCI ENV / `.pth` 等对所有入口暴露 flagtree；是否改选 `/opt/triton` 另由硬件 A/B 决定 |
| `ENV LD_LIBRARY_PATH=/opt/dtk/…`（前插，保留 base 原值） | 去掉该变量后 `from torch._C import *` → `ImportError: libgalaxyhip.so.5: cannot open shared object file: No such file or directory` | 当前 base digest 临时容器探针；commit 06ff229（把 DTK loader path 写入镜像 ENV） | FlagOS Runtime | Runtime 经 OCI ENV 或 `ld.so.conf` 提供完整 loader 合同 |
| `ENV ROCM_PATH=/opt/dtk` | 未设置时 FlagTree 选中 `/opt/rocm/llvm/bin/clang`（clang 17.0.0）：`clang (LLVM option parsing): Unknown command line argument '-check-valu-data-forward-hazards=0'`，随后 `HSACOError: File operation failed: [Errno 2] No such file or directory: '<tmp>.amdgcn'` | CI run 43426581773 / job 131895138156（E2E inference 失败）；本地探针复现同一首错误 | FlagTree / FlagOS Runtime | FlagTree 不再由该变量推导 clang，或 Runtime 自身导出 `ROCM_PATH=/opt/dtk` |
| `RUN ln -s /opt/dtk /opt/rocm` | SGLang HIP JIT builder 调用 `/opt/rocm/bin/hipcc`：`/bin/sh: 1: /opt/rocm/bin/hipcc: not found` | CI run 43417967624 / job 131868790147（E2E inference 失败）；未设 ROCM_PATH 时 FlagTree 的默认根同样是 `/opt/rocm` | SGLang（HIP JIT builder 假定 ROCm 前缀）/ FlagOS Runtime | SGLang 原生支持 DTK 前缀，或 Runtime 提供官方别名 |
| 直接 Runtime clang（wrapper 已从候选移除） | 无卡探针通过；真实 kernel 编译尚未在硬件上验证 | 上文矩阵 | 本仓库（候选选择） | Hygon 硬件矩阵通过后才声明 wrapper 可删除；失败则改用 C 或 D2 |
| 派生 xgrammar wheel `0.2.1+flagos.hcu1` | 官方 0.2.1 wheel 在 Linux/x86_64 无条件要求 standalone triton，但 HCU Runtime 的编译器能力由 `/opt/flagtree` 模块树提供且无 distribution metadata | 闭包扫描唯一声明者为 xgrammar；原 wheel 的离线 uv 对照会继续解析 triton，修正 wheel 不会；两次派生 SHA 均为 `c2deb285…1a2` 且 49 个非 RECORD 文件（含 11 个 native）逐字节不变 | xgrammar/upstream 或 FlagOS artifact owner | 发布 metadata 正确且不可变的 HCU wheel 后，删除仓库派生步骤 |
| `--excludes triton`（暂留） | 无（过渡项） | 2026-11-23 的多次 `no_excludes_gate.sh` 重试均停在 `flagos-pypi-hygon/simple/torch/` HTTP 500；离线门已通过，但按合同不足以授权删除 | 本仓库 | `no_excludes_gate.sh` 通过 → 删除后候选构建 → pip check → 完整 `verify_image.sh`，再单独删除 excludes 与精确豁免 |
| HCU patch（折入派生 wheel `0.5.18+flagos.hcu1`，输入在 `tools/wheels/hygon/`） | `ModuleNotFoundError: No module named 'aiter'`（SGLang 把一切 HIP 当 AMD ROCm，eager 导入 AMD AITER） | commit a5d492a | SGLang upstream | 上游 wheel 自带 HCU-vs-ROCm gating（届时删除派生步骤，改钉上游 wheel） |
| 插件 import guard（不伪造 `sgl_kernel.scalar_type`） | `ModuleNotFoundError: No module named 'sgl_kernel.kvcacheio'`；伪造 scalar_type 后变成 `'function' object has no attribute 'uint4b8'` | CI run 43311162022 / job 131508435886 与 run 43361932722 / job 131677473082（均 functional 失败）；时序实证见上节 | SGLang / HCU kernel wheel owner | SGLang 改为 lazy/guarded import，或 HCU 提供经验证的真实 wheel |

CI 侧（`.github/scripts/hygon/`）：测试 runner 依赖单源于
`test-requirements.txt`；容器启动快照与全部安装后 Python 校验集中在
`verify_environment.py`（`--capture-runtime-snapshot` 采集，`--plugin-source
wheel --runtime-evidence` 校验）。
