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
  -t harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl:kunlunxin-0.5.18-202611121806 .
```

In-image layout:

```text
/opt/sglang              # patched SGLang 0.5.18 sources
/opt/sglang-plugin-fl    # adapted sglang-plugin-fl sources
/opt/FlagCX              # FLAGCX_PATH
```

## Run

```bash
docker run -itd --name kunlunxin-sglang-ci-verify \
  --network=host --ipc=host --privileged --shm-size=512g \
  -v /data/models:/data/models \
  -v /data/models/Qwen:/models \
  -e SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
  harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl:kunlunxin-0.5.18-202611121806 bash

docker exec -it kunlunxin-sglang-ci-verify bash
```

After each `docker exec`, verify the environment from the sglang-plugin-fl
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
| SGLang | 0.5.18 |
| Torch | 2.9.0 |
| FlagTree | 0.7.1a1+xpu3.6 |
| FlagGems | master (commit 75823ed) |
| FlagCX | main (commit 9e42db36) |
| sglang-kernel | 0.4.2.post2+d4466f4499 |

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
`harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl@sha256:f3764ee03d4d6ce3883b952a6ea030792629de74fe785e2775fa9d5f7ca3a7c7`.

To reproduce, build `harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl:kunlunxin-0.5.18-202611121806` as above, then:

```bash
docker build -f docker/kunlunxin/ci-0.5.18.containerfile \
  -t harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl:0.5.18-kunlunxin-ci-202611121808 .
```