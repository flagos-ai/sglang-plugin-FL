# Kunlunxin P800: SGLang 0.5.18 empty / CI environment

[empty-0.5.18.containerfile](empty-0.5.18.containerfile) rebuilds on the FlagOS
Kunlunxin empty image while keeping the base Torch-2.9.0, FlagTree-0.7.1a1+xpu3.6,
FlagGems-0.0+g7fcdbed07, and FlagCX-master stack. Set the environment variables
below before running; check them both on the host and inside the container when
executing Python.

## Build

Build from the repository root on a Kunlunxin host (enter the sglang-plugin-FL
project directory). Override the base image or pip index with `BASE_IMAGE` /
`PIP_INDEX_URL` if needed.

```bash
docker build --network=host \
  -f docker/kunlunxin/empty-0.5.18.containerfile \
  -t sglang-fl:kunlunxin-0.5.18-202611021836 .
```

In-image layout:

```text
/sgl-workspace/sglang-upgrade/sglang              # patched SGLang 0.5.18 sources
/sgl-workspace/sglang-upgrade/sglang-plugin-FL    # adapted sglang-plugin-FL sources
/sgl-workspace/sglang-upgrade/FlagCX              # FLAGCX_PATH
```

## Run

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

After each `docker exec`, verify the environment from the sglang-plugin-FL
project directory:

```bash
python3 .github/scripts/kunlunxin/verify_environment.py \
    --check-versions \
    --require-gpu --min-gpus 4
```

`--require-gpu` checks that enough accelerators are visible.

Versions retained and validated on the empty base:

| Component | Version |
|---|---|
| Python | 3.10.12 |
| SGLang | 0.5.18 (with the kunlunxin source patch in this directory) |
| Torch | 2.9.0 |
| FlagTree | 0.7.1a1+xpu3.6 |
| FlagGems | master (commit 7fcdbed) |
| FlagCX | main (commit 7812d2e9) |
| sglang-kernel | 0.4.2.post2+d4466f4499 |

## Apply the SGLang source patch before install

Apply [`sglang-v0.5.18-kunlunxin.patch`](sglang-v0.5.18-kunlunxin.patch) to
**SGLang v0.5.18 sources** before installing SGLang. Installing or updating the
plugin alone does not fix the upstream hard imports.

The container build owns this install order. For a manual from-source deploy,
run the following from the SGLang tree before installing SGLang:

```bash
git apply --check "docker/kunlunxin/sglang-v0.5.18-kunlunxin.patch"
git apply "docker/kunlunxin/sglang-v0.5.18-kunlunxin.patch"
```

Then install that SGLang tree and the plugin as in the container build. The
patch only makes three hard imports optional on Kunlunxin.

## Model acceptance

```bash
pip install pytest pyyaml requests openai
# Runs unit, functional, benchmark, and e2e in order.
python tests/run.py --platform kunlunxin --device p800
```

## GitHub CI

Kunlunxin shares the unit / functional / benchmark / e2e workflows. See
[kunlunxin.yml](../../.github/configs/kunlunxin.yml). Model trees are mounted
from the host `/data/models` to `/data/models`.

Published CI tag:
`harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl:0.5.18-kunlunxin-ci-202611021838`.

To reproduce, build `sglang-fl:kunlunxin-0.5.18` as above, then:

```bash
docker build -f docker/kunlunxin/ci-0.5.18.containerfile \
  -t harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl:0.5.18-kunlunxin-ci-202611021838 .
```