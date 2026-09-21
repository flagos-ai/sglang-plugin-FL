# 华为 Ascend 910C + SGLang v0.5.18 使用与验收指南

本文档对应 `sglang-plugin-FL` 的 `dev/0.5.18` 开发线，适用于华为
Ascend 910C、CANN 8.5 和 SGLang v0.5.18 的 empty runtime。适配以官方
SGLang v0.5.18 源码为基线，当前验收模型为 Qwen3.6-27B 和
Qwen3.6-35B-A3B。本版本为基于官方 v0.5.18 和用户提供的 910C 环境说明
完成的独立适配，不依赖其它实验实现。

> **验收状态（2026-09-21）：待真机验收。** 当前开发机没有 910C，也没有
> 可用的 Docker 环境，因此本文没有填写或沿用任何 0.5.18 性能数字。
> 镜像构建、环境校验、examples 和压测入口已经固化；必须在 910C 上完成
> 本文的验收矩阵并保存原始产物后，才能将状态改为“通过”。

## 1. 适用范围与版本矩阵

推荐验收拓扑为两台主机、每台 4 张可见 910C，共 8 张卡；单机用例使用
其中一台的 4 张卡，双机用例使用 TP=4、PP=2、`nnodes=2`。

| 项目 | 固定版本或配置 |
| --- | --- |
| 架构 | Linux aarch64 |
| 操作系统参考 | Ubuntu 22.04.5 LTS，kernel 4.19.90-2102.2.0.0068.3.ctl2.aarch64 |
| NPU | Ascend 910C；单机 4 卡，双机共 8 卡 |
| 驱动 | 25.5.0 |
| CANN | 8.5.0 |
| Python | 3.11.14（校验器要求 Python 3.11） |
| PyTorch | 2.8.0 |
| torch-npu | 2.8.0.post2 |
| SGLang | v0.5.18，commit `71de97b264b04dcd514cf904003028aefe9775c8` |
| transformers | 5.12.1 |
| Triton | 3.5.0 |
| triton-ascend | 3.2.0 |
| FlagGems | 5.3.0，commit `98fae44cdf2898f39c7f24f080d7c88b83d7c593` |
| FlagCX | commit `68f069fe4ff2af9e8017b74aee8dee60c59e3b1d` |
| sgl-kernel-npu | 2026.5.1，release `2026.05.01.post2` |
| sgl-kernel-npu 压缩包 SHA-256 | `1c446e04b23497b97089591713637d512ed16135d0d97d71b6cf5c7db6787fc5` |
| 插件代码 | `flagos-ai/sglang-plugin-FL` 的 `dev/0.5.18` 分支 |

基础 empty 镜像为：

```text
harbor.baai.ac.cn/flagos-inner-models-release/flagrelease-qwen3.6-ascend-empty-tree_none-gems_5.3.0rc2-sgl_0.5.11-plugin_0.1.0-cx_0.13.0-python_3.11.14-torch_npu_2.8.0.post2-pcp_cann8.5.0-gpu_a3-arc_arm64-driver_25.5.0:202608291915
```

0.5.18 CI 镜像的目标标签为：

```text
harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl:0.2.0-ascend-sglang0.5.18-ci
```

该标签是 CI 配置使用的发布目标，不代表本次本地开发已经完成构建或推送。
首次启用 CI 前，镜像维护者必须实际构建、在 910C 上校验、推送，并将
CI 配置改成仓库返回的真实 digest；不得手写或猜测 digest。

## 2. 本版本的仓库入口

| 用途 | 文件 |
| --- | --- |
| 0.5.18 Ascend 镜像 | [`docker/ascend/empty-0.5.18.containerfile`](../docker/ascend/empty-0.5.18.containerfile) |
| 完整环境校验 | [`.github/scripts/ascend/verify_environment.py`](../.github/scripts/ascend/verify_environment.py) |
| 算子策略 | [`sglang_fl/dispatch/config/ascend.yaml`](../sglang_fl/dispatch/config/ascend.yaml) |
| 单机验收总入口 | [`scripts/ascend/run_single_node_acceptance.sh`](../scripts/ascend/run_single_node_acceptance.sh) |
| 双机 examples 总入口 | [`scripts/ascend/run_multinode_examples.sh`](../scripts/ascend/run_multinode_examples.sh) |
| 双机压测总入口 | [`scripts/ascend/run_multinode_benchmark.sh`](../scripts/ascend/run_multinode_benchmark.sh) |
| 压测驱动 | [`benchmarks/benchmark_throughput_serve.py`](../benchmarks/benchmark_throughput_serve.py) |
| example 说明 | [`examples/README.md`](../examples/README.md) |
| Ascend 测试矩阵 | [`tests/platforms/ascend.yaml`](../tests/platforms/ascend.yaml) |
| Ascend CI 容器配置 | [`.github/configs/ascend.yml`](../.github/configs/ascend.yml) |
| CI 总入口 | [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) |

## 3. 构建 0.5.18 镜像

### 3.1 前提

- 在 Linux aarch64 构建机上启用 Docker BuildKit；
- 构建机可以拉取基础镜像并访问源码和 wheel 下载地址；
- 不要设置全局 `http_proxy` 或 `https_proxy`。只有 FlagCX 的 git 操作需要
  代理时，才通过 BuildKit secret 传入单次 git proxy；
- 从仓库根目录执行构建，因为 Dockerfile 会复制插件源码和环境校验器。

### 3.2 构建 CI 镜像

```bash
git clone https://github.com/flagos-ai/sglang-plugin-FL.git
cd sglang-plugin-FL
git checkout dev/0.5.18

export ASCEND_CI_IMAGE='harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl:0.2.0-ascend-sglang0.5.18-ci'

DOCKER_BUILDKIT=1 docker build \
  --platform linux/arm64 \
  --target ci \
  -f docker/ascend/empty-0.5.18.containerfile \
  -t "${ASCEND_CI_IMAGE}" \
  .
```

如 FlagCX 的 git 拉取必须走代理，使用临时 secret，不把凭据写进
Dockerfile、镜像层或 git 全局配置：

```bash
export GIT_PROXY='http://USER:PASSWORD@HOST:PORT'
DOCKER_BUILDKIT=1 docker build \
  --platform linux/arm64 \
  --target ci \
  --secret id=git_proxy,env=GIT_PROXY \
  -f docker/ascend/empty-0.5.18.containerfile \
  -t "${ASCEND_CI_IMAGE}" \
  .
unset GIT_PROXY
```

镜像构建会完成以下工作：

1. 从固定 empty 基础镜像保留 CANN 8.5、torch 2.8 和 torch-npu；
2. 卸载旧 SGLang，并从官方 commit 安装 v0.5.18 的 `srt_empty`；
3. 安装固定版本的 transformers、xgrammar 和 compressed-tensors；
4. 安装固定 FlagGems commit；
5. 校验 NPU kernel 压缩包 SHA-256，并安装其中三个 aarch64 wheel；
6. 从固定 commit 构建 FlagCX；
7. 安装本仓库插件并执行不依赖真实 NPU 的静态环境校验；
8. 在 `ci` stage 安装固定 pytest 及 CI 工具。

### 3.3 推送并固定真实 digest

只有真机环境校验通过后才推送：

```bash
docker push "${ASCEND_CI_IMAGE}"
docker pull "${ASCEND_CI_IMAGE}"
docker inspect --format '{{index .RepoDigests 0}}' "${ASCEND_CI_IMAGE}"
```

将最后一条命令返回的完整 `repository@sha256:...` 写入
`.github/configs/ascend.yml` 的 `ci_image`。如果镜像尚未存在，应保持 CI
为待验收状态，不能以一个虚构 digest 代替。完成固定后，再把
`.github/configs/platforms.yml` 中 `ascend.enabled` 改为 `true`；仓库初始交付
保持 `false`，避免 CI 拉取尚未发布的 tag。

## 4. 启动 910C 容器

下面命令适用于每台暴露 4 张 910C 的主机。按现场路径修改 `REPO_DIR` 和
`MODEL_DIR`；双机必须使用相同代码、镜像和模型内容。

```bash
export ASCEND_CI_IMAGE='harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl:0.2.0-ascend-sglang0.5.18-ci'
export REPO_DIR=/path/to/sglang-plugin-FL
export MODEL_DIR=/path/to/models
export RESULT_DIR=/path/to/ascend-0518-results
mkdir -p "${RESULT_DIR}"

docker run --rm -it \
  --name sglang-fl-ascend-0518 \
  --init \
  --network=host \
  --ipc=host \
  --shm-size=512g \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /var/queue_schedule:/var/queue_schedule \
  -v "${REPO_DIR}":/workspace/sglang-plugin-FL \
  -v "${MODEL_DIR}":/models:ro \
  -v "${RESULT_DIR}":/results \
  -w /workspace/sglang-plugin-FL \
  --entrypoint bash \
  "${ASCEND_CI_IMAGE}"
```

如果现场设备节点或驱动安装路径不同，应按实际情况调整，但四张
`davinciN`、管理节点、驱动库和 queue scheduler 都必须在容器内可用。
不要通过隐藏缺失节点来绕过校验。

## 5. 安装当前插件与环境校验

镜像已经包含一份构建时插件。真机验收应将待测 checkout 挂载进容器，并
以 editable 方式覆盖它；`--no-deps` 防止 pip 替换厂商 torch/CANN 依赖：

```bash
cd /workspace/sglang-plugin-FL
git status --short
git rev-parse HEAD

python3 -m pip install \
  --no-deps \
  --no-build-isolation \
  -e .

export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
export SGLANG_PLUGINS=sglang_fl
```

随后运行完整校验：

```bash
python3 .github/scripts/ascend/verify_environment.py \
  --require-ci \
  --require-npu \
  --min-npus 4 \
  --plugin-root "$(pwd)"
```

校验器会检查：

- Python 和全部固定 distribution 版本；
- SGLang、FlagGems、FlagCX 的源码 commit marker；
- CANN 8.5；
- SGLang 是否仍被旧 `/sgl-workspace/sglang` editable 安装遮蔽；
- `sgl-kernel-npu` 是否具有 v0.5.18 所需模块；
- FlagCX 动态库和 Python wrapper；
- `torch.npu` 可用性、可见卡数和实际 NPU tensor 运算；
- 当前 `sglang_fl` 是否确实从挂载的 checkout 导入。

最后必须出现：

```text
[ascend-env] environment verification passed
```

任何一项不满足都应停止验收，不应在不一致的环境上继续跑 examples 或压测。

## 6. 算子策略

默认策略由 `sglang_fl/dispatch/config/ascend.yaml` 自动加载：

| SGLang 融合算子 | 后端优先级 |
| --- | --- |
| `silu_and_mul` | `flagos` → `vendor` → `reference` |
| `mrotary_embedding` | `flagos` → `vendor` → `reference` |
| `topk` | `vendor` → `flagos` → `reference` |
| `gemma_rms_norm` | `vendor` → `flagos` → `reference` |
| `fused_moe` | `vendor` → `flagos` → `reference` |
| `chunk_gated_delta_rule` | `vendor` → `flagos` → `reference` |

Qwen3.6 在 SGLang v0.5.18 的 NPU decode 热路径使用厂商原生的 fused sigmoid
recurrent update。GDN prefill 的内部 `gdn_triton.chunk_gated_delta_rule` 别名也
保留 SGLang 的 NPU wheel 直连路径，因为它返回 final state 供主 cache 回写；
这与公共 chunk API 的返回契约不同，不能用同一个 bridge 覆盖。

`fused_recurrent_gated_delta_rule` 不进入 Ascend FL dispatch。固定的 FlagGems
v5.3.0 实现采用 vLLM 风格的 inplace K×V state，而 SGLang v0.5.18 公共函数
采用 V×K/output-final-state 契约；仅替换关键字会造成错误。该函数因此保持
SGLang 原生实现，不能仅为了匹配旧配置而强制通用 FlagGems kernel。

通常无需设置 `SGLANG_FL_PER_OP`。需要复现实验或定位问题时，可显式覆盖：

```bash
export SGLANG_FL_PER_OP='silu_and_mul=flagos;mrotary_embedding=flagos;topk=vendor;gemma_rms_norm=vendor;fused_moe=vendor;chunk_gated_delta_rule=vendor'
```

验收脚本会设置插件、FlagGems、FlagCX 和运行模式所需环境，并保留调用者
已经显式设置的值。正式提交结果时，应在 `environment.txt` 中同时记录任何
覆盖，避免不同策略的结果被混在一起。

### 6.1 正确性模式

单机和双机 examples 使用正确性模式：

```text
SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
HCCL_BUFFSIZE=2400
```

不要设置 `ASCEND_LAUNCH_BLOCKING=1`。同步 launch 会改变编译时机，并可能
触发动态 kernel 编译失败。

### 6.2 性能模式

双机压测入口默认使用：

```text
SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
SGLANG_NPU_USE_MULTI_STREAM=1
STREAMS_PER_DEVICE=32
HCCL_BUFFSIZE=1000
HCCL_OP_EXPANSION_MODE=AIV
```

这些设置只用于性能验收；不要用性能模式替代前面的正确性矩阵。

## 7. 测试模型和图片

默认模型目录为：

```text
/models/Qwen3.6-27B
/models/Qwen3.6-35B-A3B
```

可分别通过 `MODEL_27B_PATH` 和 `MODEL_35B_PATH` 覆盖。模型必须提前放到
两台机器的对应路径，验收脚本不会联网下载模型。

离线、多模态并发和双机 examples 要求 `IMAGE_DIR` 下存在以下四个非空文件：

```text
red_square.jpg
cat.jpg
stop_sign.png
digit_seven.png
```

仓库默认使用 `examples/test_images`。缺少模型、目录、图片或图片为空都会
直接失败，不会静默跳过视觉用例。

## 8. 单机 4 卡 examples 验收

### 8.1 一键执行完整矩阵

在一台可见 4 张 910C 的容器内执行。任务耗时较长，必须使用 `tmux`：

```bash
cd /workspace/sglang-plugin-FL
mkdir -p /results/ascend-0518

tmux new-session -d -s ascend-single \
  'MODEL_27B_PATH=/models/Qwen3.6-27B \
   MODEL_35B_PATH=/models/Qwen3.6-35B-A3B \
   bash scripts/ascend/run_single_node_acceptance.sh \
     --result-dir /results/ascend-0518/single'
```

查看进度和会话状态：

```bash
tmux capture-pane -t ascend-single -p
tmux has-session -t ascend-single 2>/dev/null && echo running || echo done
```

默认矩阵为：

| 模型 | 用例 | 并行配置 | 次数 |
| --- | --- | --- | ---: |
| Qwen3.6-27B | offline 文本 + VL | TP=4 | 1 |
| Qwen3.6-27B | concurrent `--mode all`：text + VL + mixed | TP=4 | 1 |
| Qwen3.6-27B | concurrent `--mode text` canary | TP=2 | 3 次独立建引擎 |
| Qwen3.6-27B | MTP 与 baseline 正确性比较 | TP=4 | 1 |
| Qwen3.6-35B-A3B | offline 文本 + VL | TP=4 | 1 |
| Qwen3.6-35B-A3B | concurrent `--mode all`：text + VL + mixed | TP=4 | 1 |
| Qwen3.6-35B-A3B | concurrent `--mode text` canary | TP=2 | 3 次独立建引擎 |

只跑一个模型时可加 `--model 27b` 或 `--model 35b`；完整验收不得使用该选项。

### 8.2 单独运行 example

需要定位某一模型时，可直接运行仓库脚本：

```bash
MODEL_PATH=/models/Qwen3.6-27B TP_SIZE=4 \
python3 examples/qwen3_6_27b_offline_inference.py

MODEL_PATH=/models/Qwen3.6-27B TP_SIZE=4 \
python3 examples/qwen3_6_27b_concurrent.py --mode all

MODEL_PATH=/models/Qwen3.6-35B-A3B TP_SIZE=4 \
python3 examples/qwen3_6_35b_a3b_offline_inference.py

MODEL_PATH=/models/Qwen3.6-35B-A3B TP_SIZE=4 \
python3 examples/qwen3_6_35b_a3b_concurrent.py --mode all
```

单独命令适合排障，但不能代替总入口对模型、图片、NPU 数量、重复 canary
和产物的强校验。

### 8.3 TP=2 文本并发 canary

Ascend 算子与 FlagGems Ascend 算子混用时，TP=2 文本并发曾出现非稳定问题。
因此本版本把它保留为强制 canary，而不是标记为“预期失败”。总入口对两个
模型分别重建引擎运行 3 次；任何一次返回非零都会使整套验收失败。

独立复现命令为：

```bash
MODEL_PATH=/models/Qwen3.6-27B TP_SIZE=2 \
python3 examples/qwen3_6_27b_concurrent.py --mode text

MODEL_PATH=/models/Qwen3.6-35B-A3B TP_SIZE=2 \
python3 examples/qwen3_6_35b_a3b_concurrent.py --mode text
```

若失败，应保留对应 `*_tp2_canary_*.log`，不得只重跑到偶然成功后删除失败
记录。

### 8.4 单机通过判据

- 所有脚本退出码为 0；
- 没有用例因模型或图片缺失而跳过；
- `single/environment.txt` 记录的是本次 checkout 和固定依赖；
- 每个阶段日志存在且非空；
- 末行出现 `PASS: single-node acceptance completed; no command was skipped`。

## 9. 双机 TP=4 + PP=2 examples 验收

两台主机都要进入同版本容器，准备相同模型和图片，并保证业务网卡互通。
以下示例假设 master 业务 IP 是 `172.16.10.108`，HCCL/Gloo 网卡名为
`business`；必须替换成现场值。两端命令应尽量同时启动。

master 节点：

```bash
cd /workspace/sglang-plugin-FL
tmux new-session -d -s ascend-multinode-examples \
  'bash scripts/ascend/run_multinode_examples.sh \
    --role master \
    --master-addr 172.16.10.108 \
    --interface business \
    --model all \
    --result-dir /results/ascend-0518/examples-master'
```

worker 节点：

```bash
cd /workspace/sglang-plugin-FL
tmux new-session -d -s ascend-multinode-examples \
  'bash scripts/ascend/run_multinode_examples.sh \
    --role worker \
    --master-addr 172.16.10.108 \
    --interface business \
    --model all \
    --result-dir /results/ascend-0518/examples-worker'
```

查看日志：

```bash
tmux capture-pane -t ascend-multinode-examples -p
```

入口固定 `tp_size=4`、`pp_size=2`、`nnodes=2`，依次验证 27B 和
35B-A3B 的文本、并发文本和视觉请求。第二个模型使用与第一个模型错开的
API、rendezvous 和 collective 端口。

SGLang v0.5.18 的非零 rank 在全部 scheduler 正常退出后可能出现一个已知
HTTP shutdown `rc=3`。worker wrapper 只在完整日志严格匹配该已知顺序时将
其归一化为成功；任意其它 `rc=3`、scheduler 未完整退出或异常 traceback
仍然失败。

双机通过要求两端都出现各自的最终 `PASS`，并保存两端
`environment.txt` 与 `*_master.log` / `*_worker.log`。仅 master 成功不算通过。

## 10. 双机 serving 压测

### 10.1 固定矩阵

两个模型分别执行下列矩阵：

| 输入 token | 输出 token | 请求数 | 最大并发 | 轮数 | 汇总规则 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1,024 | 1,024 | 64 | 64 | 4 | 第 1 轮预热，平均第 2–4 轮 |
| 4,096 | 1,024 | 64 | 64 | 4 | 第 1 轮预热，平均第 2–4 轮 |
| 16,384 | 1,024 | 64 | 64 | 4 | 第 1 轮预热，平均第 2–4 轮 |

每个模型应产生 12 条 raw run 和 3 条 summary；两个模型合计 24 条 raw run
和 6 条 summary。压测驱动调用官方 v0.5.18 入口
`python -m sglang.benchmark.serving`，使用 `random-ids`、固定 seed、无限请求
速率和精确长度输入。

### 10.2 启动压测

验收脚本会设置进程级性能变量，但不会擅自修改宿主机 sysctl。需要比较跨版本
性能时，应由机器管理员在两台主机上统一 CPU/NUMA 策略，并先记录原值：

```bash
for governor in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo performance | sudo tee "${governor}"
done
sudo sysctl -w vm.swappiness=0
sudo sysctl -w kernel.numa_balancing=0
sudo sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_FL_WATCHDOG_DIAG=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ASCEND_LAUNCH_BLOCKING
```

这些是宿主机级性能基线设置，不是模型正确性的前提；如果现场不允许修改，
应保持两端一致并在验收记录中注明，而不是无记录地混用不同配置。

master 节点：

```bash
cd /workspace/sglang-plugin-FL
tmux new-session -d -s ascend-benchmark \
  'bash scripts/ascend/run_multinode_benchmark.sh \
    --role master \
    --master-addr 172.16.10.108 \
    --interface business \
    --model all \
    --result-dir /results/ascend-0518/benchmark-master'
```

worker 节点：

```bash
cd /workspace/sglang-plugin-FL
tmux new-session -d -s ascend-benchmark \
  'bash scripts/ascend/run_multinode_benchmark.sh \
    --role worker \
    --master-addr 172.16.10.108 \
    --interface business \
    --model all \
    --result-dir /results/ascend-0518/benchmark-worker'
```

压测入口会启动 TP=4、PP=2 的服务；master 等待 `/health` 后运行客户端，
worker 只加入分布式服务。默认启动参数包括：

- `--attention-backend ascend`、`--device npu`、`--dtype bfloat16`；
- `--max-running-requests 64`；
- `--cuda-graph-max-bs-decode 70`；
- `--disable-radix-cache` 和 `--trust-remote-code`；
- 性能模式中的 overlap、multi-stream 和 HCCL 设置。

如已有服务，也可以只运行驱动：

```bash
python3 benchmarks/benchmark_throughput_serve.py \
  --model /models/Qwen3.6-35B-A3B \
  --model-name qwen3_6_35b_a3b \
  --host 127.0.0.1 \
  --port 30000 \
  --output-dir /results/ascend-0518/manual-benchmark
```

### 10.3 压测的严格通过条件

每一轮都必须同时满足：

- 子进程退出码为 0；
- official JSONL 恰好一条记录；
- `completed == 64`，没有请求错误；
- 每个请求的 input/output 长度分别等于本组目标；
- 总输入 token 等于 `input_len × 64`；
- 总输出 token 等于 `1024 × 64`；
- 所有必需的吞吐和时延指标存在且为有限数值。

任一轮失败都会使最终进程返回非零，并写入 `failures.txt`。不能用另外三轮
平均值掩盖失败轮次。完整成功时末行是：

```text
PASS: all 12 runs completed with exact request and token counts
```

### 10.4 压测产物

每个模型的时间戳目录包含：

```text
configuration.json
official-jsonl/
  1024_1024_c64_run1.jsonl ... run4.jsonl
  4096_1024_c64_run1.jsonl ... run4.jsonl
  16384_1024_c64_run1.jsonl ... run4.jsonl
  同名 .log 文件
raw_runs.csv
summary.csv
failures.txt                 # 仅失败时存在
```

`raw_runs.csv` 应有 12 条数据行；`summary.csv` 应有 3 条数据行，且只平均
第 2–4 轮。发布性能结论时必须同时归档 JSONL、客户端日志、server 日志、
CSV、`configuration.json` 和两端 `environment.txt`，不能只抄一张汇总表。

## 11. CI 配置

Ascend 工作流和测试矩阵已经接线，但 `.github/configs/platforms.yml` 当前保持
`enabled: false`。完成 ARM64 镜像构建、910C 验收与镜像推送并固定真实 digest
后，再启用 Ascend；启用后 CI 会在 `dev/0.5.18` 的 push 和 pull request 上运行。
Ascend 配置要求 runner 具有以下标签：

```text
self-hosted
Linux
ARM64
flagcicd-910c
```

Runner 宿主机必须提供 4 张 NPU、驱动/固件/queue scheduler 挂载，以及预先
放置在 `/mnt/airs-business/cicd/models` 的离线模型。CI 不下载模型。

流水线顺序为：

1. `check.sh` 输出设备诊断，并用完整校验器要求至少 4 张工作 NPU；
2. `setup.sh` 以 `--no-deps` 安装当前 checkout，再次校验导入路径；
3. unit tests；
4. functional tests；
5. 27B/35B-A3B inference、serving、concurrent E2E，Ascend 固定串行执行；
6. throughput、latency、serve benchmark smoke。

CI 的 benchmark job 是入口级 smoke test，不等于第 10 节的双机固定长度性能
验收。发布 0.5.18 时两者都要通过。

在镜像尚未推送或仍使用可变 tag 时，CI 仍是“待部署”状态。建议的启用顺序
是：构建镜像 → 910C 环境校验 → 推送 → 固定真实 digest → 触发 CI → 保存
CI 链接和 artifacts。

## 12. 最终交付产物清单

完成验收后应归档：

- 插件 commit SHA 和 `git status --short`；
- 镜像完整 `repository@sha256:...`；
- 每台主机的 `npu-smi info`；
- 环境校验器完整输出；
- 单机两模型 TP=4 offline/concurrent 日志；
- 两模型各 3 次 TP=2 canary 日志；
- 双机 examples 的 master/worker 日志；
- 双机 benchmark 的 master/worker server 日志；
- 两模型全部 JSONL、raw CSV、summary CSV 和 configuration；
- CI run 链接及 unit、functional、E2E、benchmark artifacts；
- 异常场景的失败日志，不得删除后只保留重试成功结果。

## 13. 常见故障排查

### 13.1 仍然导入旧 SGLang

症状是校验器报告来自 `/sgl-workspace/sglang`，或发现旧 editable `.pth`。

```bash
python3 - <<'PY'
import sglang
print(sglang.__file__)
PY
python3 -m pip show sglang
```

不要手工改 `PYTHONPATH` 掩盖问题。重新用 0.5.18 Dockerfile 构建镜像；该
Dockerfile 会先卸载旧包，校验器也会阻止旧 checkout 抢占导入。

### 13.2 NPU 不可见或少于 4 张

检查容器内 `/dev/davinci0..3`、`/dev/davinci_manager`、`/dev/devmm_svm`、
`/dev/hisi_hdc`，以及 driver bind mount。`npu-smi` 只是诊断工具；最终以
`torch.npu.is_available()`、device count 和实际 tensor probe 为准。

### 13.3 CANN 或 Python 包版本不一致

直接运行环境校验器，不要通过放宽版本比较继续验收。常见原因是 `pip install`
没有使用 `--no-deps`，导致 torch、transformers 或 Triton 被替换。

### 13.4 NPU kernel 模块缺失

必须使用表中固定的 aarch64 kernel 压缩包并核对 SHA-256。不要用空 Python
stub 伪装缺失的 native module；镜像构建和真机导入检查都应失败。

### 13.5 FlagCX 加载失败

默认镜像路径是 `/opt/FlagCX`。确认：

```bash
echo "${FLAGCX_PATH}"
test -f "${FLAGCX_PATH}/build/lib/libflagcx.so"
test -f "${FLAGCX_PATH}/plugin/interservice/flagcx_wrapper.py"
```

同时确认两台主机的 `HCCL_SOCKET_IFNAME`、`GLOO_SOCKET_IFNAME` 和
`NCCL_SOCKET_IFNAME` 指向同一条可互通业务网，并检查 rendezvous/API/
collective 端口没有被占用或防火墙阻断。

### 13.6 图片用例被跳过或直接失败

四张 fixture 都必须存在且非空。总验收脚本会在创建引擎前检查，因此应修复
`IMAGE_DIR` 或挂载路径，不应修改脚本让用例跳过。

### 13.7 TP=2 文本并发偶发错误

这是 canary 要捕获的问题。保留失败轮次的环境、算子策略和完整日志，可临时
比较 `SGLANG_FL_PER_OP` 的后端选择来定位，但不要把排障配置混入正式结果，
也不要将偶发失败标成预期通过。

### 13.8 动态 kernel 编译失败

先确认没有设置 `ASCEND_LAUNCH_BLOCKING`，再核对 CANN、torch-npu、Triton
和 kernel wheel 是否完全匹配版本矩阵。不要用全局同步模式规避异步问题。

### 13.9 benchmark 产生 `failures.txt`

从对应 `.log` 和 official JSONL 检查服务错误、请求错误和实际 token 长度。
本驱动有意拒绝部分成功、少 token 和缺指标结果；不要编辑 CSV 伪造通过。

## 14. 0.5.18 验收记录

本文初始状态如下，完成真机运行后由验收人填写实际日期、commit、digest 和
产物位置：

| 项目 | 当前状态 | 通过证据 |
| --- | --- | --- |
| aarch64 CI 镜像构建 | 待真机/构建机验收 | 镜像构建日志 + digest |
| 4 卡环境校验 | 待真机验收 | verifier 完整日志 |
| 27B 单机 TP=4 examples | 待真机验收 | offline/concurrent/MTP 日志 |
| 35B-A3B 单机 TP=4 examples | 待真机验收 | offline/concurrent 日志 |
| 两模型 TP=2 canary，各 3 次 | 待真机验收 | 6 份 canary 日志 |
| 两模型双机 TP=4 + PP=2 examples | 待真机验收 | 两端日志与 PASS 标记 |
| 27B 双机固定矩阵压测 | 待真机验收 | 12 JSONL + raw/summary CSV |
| 35B-A3B 双机固定矩阵压测 | 待真机验收 | 12 JSONL + raw/summary CSV |
| Ascend CI 全链路 | 待镜像发布后验收 | CI run 链接与 artifacts |

在上述项目全部有可追溯证据之前，本适配只能描述为“代码与验收入口已准备”，
不能描述为“910C 0.5.18 已完成性能验收”。
