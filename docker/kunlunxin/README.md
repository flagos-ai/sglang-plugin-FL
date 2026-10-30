# 昆仑芯 P800：SGLang 0.5.18 empty / CI 环境说明

[empty-0.5.18.containerfile](empty-0.5.18.containerfile)基于 FlagOS 昆仑芯 empty 镜像，保留基底上的
Torch-2.9.0、FlagTree-0.7.1a1+xpu3.6、FlagGems-0.0+g7fcdbed07 与 FlagCX-master。运行前需要设置以下环境变量；
在宿主机直跑和进入容器后执行 Python 时都应检查这些设置。

## 构建

在昆仑芯宿主机的仓库根目录构建。进入到sglang-plugin-fl项目目录，可通过 `BASE_IMAGE` / `PIP_INDEX_URL` 覆盖默认基底与 pip 源。

```bash
docker build --network=host \
  -f docker/kunlunxin/empty-0.5.18.containerfile \
  -t sglang-fl:kunlunxin-0.5.18 .
```

镜像内路径约定：

```text
/sgl-workspace/sglang-upgrade/sglang              # 补丁后的 SGLang 0.5.18 源码
/sgl-workspace/sglang-upgrade/sglang-plugin-FL    # 适配后的 sglang-plugin-FL 源码
/sgl-workspace/sglang-upgrade/FlagCX              # FLAGCX_PATH
```

## 启动
```bash
docker run -itd --name kunlunxin-sglang-ci-verify \
  --network=host --ipc=host --privileged --shm-size=512g \
  -v /data/models:/data/models \
  -v /data/models/Qwen:/models \
  -e USE_FLAGGEMS=1 -e SGLANG_PLUGINS=sglang_fl \
  -e SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
  -e SGLANG_FL_DIST_BACKEND=flagcx \
  -e FLAGCX_PATH=/sgl-workspace/sglang-upgrade/FlagCX \
  sglang-fl:kunlunxin-0.5.18 bash

docker exec -it kunlunxin-sglang-ci-verify bash
```

每次 `docker exec` 执行Python时也应检查环境：进入到sglang-plugin-fl项目目录

```bash
python3 .github/scripts/kunlunxin/verify_environment.py \
    --check-versions \
    --require-gpu --min-gpus 4
```
`--require-gpu` 确定加速卡满足要求。

已在原始 empty 基底上保留并验证的版本：

| 组件 | 海光版本 |
|---|---|
| Python | 3.10.12 |
| SGLang | 0.5.18（应用本目录 kunlunxin 源码补丁） |
| Torch | 2.9.0 |
| FlagTree | 0.7.1a1+xpu3.6 |
| FlagGems | master(commit 7fcdbed) |
| FlagCX | main(commit 7812d2e9) |
| sglang-kernel | 0.4.2.post2+d4466f4499 |

## 安装前应用 SGLang 源码补丁

必须对 **SGLang v0.5.18 源码**应用
[`sglang-v0.5.18-kunlunxin.patch`](sglang-v0.5.18-kunlunxin.patch)，然后再安装 SGLang。
仅安装或更新插件无法修复上游导入问题。

容器构建负责安装顺序。手动从源码部署时，在安装 SGLang 前执行：进入sglang项目目录

```bash
git apply --check "docker/kunlunxin/sglang-v0.5.18-kunlunxin.patch"
git apply "docker/kunlunxin/sglang-v0.5.18-kunlunxin.patch"
```

之后按容器构建安装该 SGLang 源码与插件。补丁仅让昆仑芯跳过三个硬导入。

## 模型验收

```bash
pip install pytest pyyaml requests openai
# 依次运行 unit \ function \ benchmark \ e2e。
python tests/run.py --platform kunlunxin --device p800
```

## GitHub CI

昆仑芯共享unit/functional/benchmark/E2E 流程。CI 配置见 [kunlunxin.yml](../../.github/configs/kunlunxin.yml)；
模型目录从宿主机 `/data/models` 映射到 `/data/models`。

已发布的 CI tag 为`harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl@sha256:b0faa6a570501d8445aeb6fd78c5bbe58499b2573d5316a3d2902ed994f6245a`。

复现构建先按上文生成 `sglang-fl:kunlunxin-0.5.18`，再执行：

```bash
docker build -f docker/kunlunxin/ci-0.5.18.containerfile -t harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl:0.5.18-kunlunxin-ci .
```
