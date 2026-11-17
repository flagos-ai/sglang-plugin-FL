# 清微 TX8110（TXDA）：SGLang 0.5.18 empty 环境运行说明

本目录把清微 TXDA 的 SGLang 0.5.11 镜像转换为单一 Python 环境的 SGLang 0.5.18
`[srt_empty]` 运行时。厂商栈保持不变：torch 2.11.0+cpu、torch_txda、FlagTree
0.7.1a1+tsingmicro3.6、FlagGems、Kuiper runtime/CCL 0.30279.6、FlagCX。

| 文件 | 作用 |
|---|---|
| `install_sglang_0518.sh` | 卸载 SGLang 0.5.11、旧插件和 CUDA-only wheel；用厂商栈约束文件安装 SGLang 0.5.18；安装插件；最后做环境校验。containerfile 使用它，也可以在已有容器里手动执行 |
| `empty-0.5.18.containerfile` | `runtime` 和 `ci` 两个 target。`ci` 额外安装测试工具（pytest、ruff 等） |
| `verify_environment.py` | `--metadata-only` 只检查包版本和模块位置，不碰设备，可在 `docker build` 中运行；`--require-device --min-cards N` 额外检查可见卡数 |

## 构建与启动

在仓库根目录执行：

```bash
docker build -f docker/tsingmicro/empty-0.5.18.containerfile \
    --target ci -t sglang-plugin-fl:0.5.18-tsingmicro-ci .
```

- `BASE_IMAGE`：TXDA 的 SGLang 0.5.11 镜像，需带 0.30279.6 runtime/CCL。默认值是本地
  slim 镜像，发布时改为 Harbor 地址：`--build-arg BASE_IMAGE=<image>`。
- GitHub 访问慢的站点可以用镜像源提供 SGLang 源码包：
  `--build-arg SGLANG_SOURCE_URL=<url> --build-arg SGLANG_SOURCE_SHA256=<sha256>`。
  v0.5.18 源码包对应提交 `71de97b264`。
- 构建需要 `SGLANG_BUILD_RUST_EXTS=none`（安装脚本已默认设置），与 MUSA、海光相同。

启动容器不需要 `--privileged`，传入设备节点即可：

```bash
docker run -it --network host --ipc=host --shm-size=64g \
    --ulimit memlock=-1 --ulimit stack=67108864 --init \
    --device /dev/accel --device /dev/accel_drv_mgr \
    -v /mnt/nvme_data/models:/models:ro \
    sglang-plugin-fl:0.5.18-tsingmicro-ci bash
```

## 设备选择

- 用 `TXDA_VISIBLE_DEVICES` 指定卡，例如 `TXDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`。
- torch_txda 启动时的当前设备是**最后一张可见卡**，而不是 0 号卡。自己写的脚本如果在
  `txda:0` 上分配张量，先调用 `torch.txda.set_device(0)`，否则会跨卡访问，轻则
  `txMemcpyAsync ... Invalid parameters`，重则触发卡的硬件异常。
- 主机共享时，`TXDA_VISIBLE_DEVICES` 隔离不够：驱动在某张卡故障恢复时，会杀掉所有在这张卡上
  持有上下文的进程。只使用分配给自己的空闲卡组。

## 默认配置

插件在 TXDA 上自动应用下列设置，启动脚本不需要再传：

- server 参数（`vendor/tsingmicro/runtime_config.py`，只在用户没有显式指定时生效）：
  `device=txda`、`dtype=bfloat16`、关闭 radix cache、`watchdog_timeout=3600`、
  视觉 attention 用 `triton_attn`、图片处理用 PIL。
- 环境变量：`PRECISION_MODE=2`（已设置的值不会被覆盖）、`SGLANG_FL_GDN_NORM` 默认开启。
- FlagGems 黑名单：`sglang_fl/dispatch/config/tsingmicro.yaml` 的 `flagos_blacklist`，
  `SGLANG_FL_FLAGOS_BLACKLIST` 环境变量会整体覆盖它。

与模型相关的参数（`context_length`、`chunked_prefill_size` 等）仍在 examples 和测试用例里。

## 环境检查与模型验收

```bash
python3 docker/tsingmicro/verify_environment.py --plugin-root . --require-device --min-cards 8

# 单机 TP8：离线和并发
export TXDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TP_SIZE=8
python examples/qwen3_6_35b_a3b_offline_inference.py
python examples/qwen3_6_35b_a3b_concurrent.py
python examples/qwen3_6_27b_offline_inference.py
python examples/qwen3_6_27b_concurrent.py

# 与 CI 相同的测试入口
python tests/run.py --platform tsingmicro --scope e2e --stage nightly
python tests/run.py --platform tsingmicro --scope benchmark --stage nightly
```

首次运行要编译 Triton/FlagGems kernel，冷缓存下 35B 离线约 7 分钟，并发约 14-16 分钟。

## 已知问题与规避

| 问题 | 规避 |
|---|---|
| SGLang 0.5.18 的 vocab parallel embedding Triton kernel 在 TXDA 上不遵守 masked load，TP 下输出全是 `"\n"` | `patches/vocab_embedding.py` 回退到 torch 实现；`SGLANG_FL_TXDA_TRITON_EMBEDDING=1` 可重新启用 Triton kernel 做复测 |
| torch_txda 的 `uniform_` 走 CPU kernel，不接受 txda generator，`load_format=dummy` 加载模型失败 | `patches/dummy_weights.py` 在 CPU 上生成 dummy 权重再拷到设备 |
| `sglang.bench_one_batch` 的 worker 不调用 `load_plugins()`，插件不生效 | `plugin_bootstrap.py` 在每个 worker 第一次构造 `ModelConfig` 前加载插件，做法与天数相同 |
| torch_txda 的 `TXDAGraph` 捕获空图，不支持 graph | 平台声明不支持 CUDA graph，相关测试跳过 |

## 已验证范围（2026-11）

TX8110，driver/runtime/CCL 0.30279.6，Qwen3.6-35B-A3B 和 Qwen3.6-27B，TP8：

- 单机离线、并发（text、VL、mixed）：通过，回答与 SGLang 0.5.11 一致。
- 两机 TP8/PP2：两个模型各 10/10 通过。
- serving、throughput/latency/serve benchmark smoke：通过。
- 本目录 containerfile 构建的镜像上，四项单机测试冷缓存通过。

未覆盖：MTP（EAGLE）只有单元测试，没有在设备上验证；长时间稳定性和 32k 上下文 server 压测未做。

## GitHub CI

配置见 [tsingmicro.yml](../../.github/configs/tsingmicro.yml) 和
[tests/platforms/tsingmicro.yaml](../../tests/platforms/tsingmicro.yaml)。平台在
`platforms.yml` 里是 `enabled: false`，镜像发布、`ci_image` 固定 digest、runner
`ci-sglang-plugin-fl-tsingmicro` 上线之后再打开。

- 每个 job 先运行 `.github/scripts/tsingmicro/check.sh`，选出一组 8 张空闲且健康的卡，导出
  `TXDA_VISIBLE_DEVICES`。选卡依次检查：`tsm_smi` 占用、主机维护的排除列表
  （`TSINGMICRO_CI_BAD_CARDS` / `TSINGMICRO_CI_BAD_CARDS_FILE`，容器内读不到 dmesg）、
  每张卡的小张量计算探测。`TSINGMICRO_CI_CARDS` 限定候选卡范围。
- 用例全部是 TP8，`e2e_serial: true`，同一卡组上串行执行。
- pr 阶段：inference、serving；benchmark 只跑 serve。nightly 阶段再加 concurrent，benchmark
  跑 throughput、latency、serve。
- 不挂载 Triton 缓存，每次运行都重新编译，job 超时设为 240 分钟。

工作流配置完成不代表 GitHub 硬件 CI 已通过，应以实际 job 结果为准。
