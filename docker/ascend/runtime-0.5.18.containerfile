# syntax=docker/dockerfile:1.7
# SGLang 0.5.18 two-wheel CI image on the Ascend 910C operations runtime
# (CANN 9.2.1, torch_npu 2.13, FlagTree under /opt/flagtree).
# Assembly only; build and evidence are recorded in docker/ascend/README.md.
# Moving semver tag per the Runtime owner; the build records the resolved
# digest in staging-record.txt.
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-ascend-cann9.2.1-910c:2.3.0
FROM ${BASE_IMAGE}

# The public PyPI mirror the resolver uses; the Runtime configures no index
# of its own (only a trusted host for its file store). Override with
# --build-arg RUNTIME_INDEX=<url> if the mirror is unreachable from the
# build host.
ARG RUNTIME_INDEX=https://mirrors.cloud.tencent.com/pypi/simple

# One uv resolver install. The constraints and wheels are read-only bind
# mounts, so they never enter an image layer; the constraints are the
# Runtime's own `pip freeze --all` read from the exact base, so the resolver
# can never replace a Runtime distribution. No standalone triton resolves on
# this Runtime (its compiler is the /opt/flagtree module tree), and no
# exclusion list is needed. The Runtime's Bash environment (BASH_ENV) exposes
# that slot to every non-interactive bash; plugin activation happens through
# SGLang's entry points when torch_npu reports devices.
RUN --mount=type=bind,source=artifacts,target=/tmp/artifacts,readonly \
    /usr/local/bin/uv pip install --python /flagos/bin/python3 \
      --no-cache \
      --default-index "$RUNTIME_INDEX" \
      --constraints /tmp/artifacts/runtime-constraints.txt \
      '/tmp/artifacts/sglang-0.5.18-py3-none-any.whl[srt_empty,runtime_common]' \
      '/tmp/artifacts/sglang_fl-0.1.0-py3-none-any.whl[ascend]'
